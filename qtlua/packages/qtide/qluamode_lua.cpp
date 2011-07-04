/* -*- C++ -*- */

#include <QtGlobal>
#include <QtAlgorithms>
#include <QChar>
#include <QDebug>
#include <QList>
#include <QMap>
#include <QPointer>
#include <QRegExp>
#include <QSettings>
#include <QSharedData>
#include <QSharedDataPointer>


#include "qluaapplication.h"
#include "qtluaengine.h"
#include "qluamainwindow.h"
#include "qluatextedit.h"
#include "qluamode.h"

#include <string.h>
#include <ctype.h>

#define DEBUG 0


// ========================================
// USERDATA


namespace {

  enum TokenType {
    // generic tokens
    Other, Identifier, Number, String,
    // tokens
    SemiColon, ThreeDots, Comma, Dot, Colon,
    LParen, RParen, LBracket, RBracket, LBrace, RBrace, 
    // keywords
    Kand, Kfalse, Kfunction, Knil, 
    Knot, Kor, Ktrue, Kin,
    // keywords that kill statements
    Kbreak, Kdo, Kelse, Kelseif, Kend, Kfor,
    Kif, Klocal, Krepeat, Kreturn,
    Kthen, Kuntil, Kwhile,
    // special
    Eof,
    Chunk,
    Statement,
    StatementCont,
    FunctionBody,
    FunctionName,
    FirstKeyword = Kand,
    FirstStrictKeyword = Kbreak,
  };

  struct Keywords { 
    const char *text; 
    TokenType type; 
  };
  
  Keywords skeywords[] = {
    {"and", Kand}, {"break", Kbreak}, {"do", Kdo}, {"else", Kelse},
    {"elseif", Kelseif}, {"end", Kend}, {"false", Kfalse}, {"for", Kfor},
    {"function", Kfunction}, {"if", Kif}, {"in", Kin}, {"local", Klocal},
    {"nil", Knil}, {"not", Knot}, {"or", Kor}, {"repeat", Krepeat},
    {"return", Kreturn}, {"then", Kthen}, {"true", Ktrue},
    {"until", Kuntil}, {"while", Kwhile},
    {";", SemiColon}, {"...", ThreeDots}, {",", Comma}, 
    {".", Dot}, {":", Colon}, 
    {"(", LParen}, {")", RParen}, {"[", LBracket}, 
    {"]", RBracket}, {"{", LBrace}, {"}", RBrace}, 
    {0}
  };

  struct Node;

  struct PNode : public QSharedDataPointer<Node> {
    PNode();
    PNode(TokenType t, int p, int l, PNode n);
    PNode(TokenType t, int p, int l, int i, PNode n);
    TokenType type() const;
    int pos() const;
    int len() const;
    int indent() const;
    PNode next() const;
  };
  
  struct Node : public QSharedData {
    Node(TokenType t, int p, int l, PNode n)
      : next(n), type(t),pos(p),len(l),indent(-1) {}
    Node(TokenType t, int p, int l, int i, PNode n)
      : next(n), type(t),pos(p),len(l),indent(i) {}
    PNode next;
    TokenType type;
    int pos;
    int len;
    int indent;
  };
  
  PNode::PNode()
    : QSharedDataPointer<Node>() {}
  
  PNode::PNode(TokenType t, int p, int l, PNode n)
    : QSharedDataPointer<Node>(new Node(t,p,l,n)) {}
  
  PNode::PNode(TokenType t, int p, int l, int i, PNode n)
    : QSharedDataPointer<Node>(new Node(t,p,l,i,n)) {}

  inline TokenType PNode::type() const { 
    const Node *n = constData();
    return (n) ? n->type : Chunk; 
  }
  
  inline int PNode::pos() const {
    const Node *n = constData();
    return (n) ? n->pos : 0;
  }

  inline int PNode::len() const {
    const Node *n = constData();
    return (n) ? n->len : 0;
  }
  
  inline int PNode::indent() const {
    const Node *n = constData();
    return (n) ? n->indent : 0;
  }
  
  inline PNode PNode::next() const {
    const Node *n = constData();
    return (n) ? n->next : PNode();
  }
    
  struct UserData : public QLuaModeUserData
  {
    // lexical state
    int lexState;
    int lexPos;
    int lexN;
    // parser state
    PNode nodes;
    int lastPos;
    // initialize
    UserData() : lexState(0), lexPos(0), lexN(0), lastPos(0) {}
    virtual int highlightState() { return (lexState<<16)^lexN; }
  };

}




// ========================================
// QLUAMODELUA


class QLuaModeLua : public QLuaMode
{
  Q_OBJECT
public:
  QLuaModeLua(QLuaTextEditModeFactory *f, QLuaTextEdit *e);
  void gotLine(UserData *d, int pos, int len, QString);
  void gotToken(UserData *d, int pos, int len, QString, TokenType);
  bool supportsComplete() { return true; }
  bool supportsLua() { return true; }
  virtual void parseBlock(int pos, const QTextBlock &block, 
                          const QLuaModeUserData *idata, 
                          QLuaModeUserData *&odata );
  QStringList computeFileCompletions(QString s, bool escape, QString &stem);
  QStringList computeSymbolCompletions(QString s, QString &stem);
  virtual bool doComplete();
private:
  QMap<QString,TokenType> keywords;
  QRegExp reNum, reSym, reId;
  int bi;
};


QLuaModeLua::QLuaModeLua(QLuaTextEditModeFactory *f, QLuaTextEdit *e)
  : QLuaMode(f,e),
    reNum("^(0x[0-9a-fA-F]+|\\.[0-9]+|[0-9]+(\\.[0-9]*)?([Ee][-+]?[0-9]*)?)"),
    reSym("^(\\.\\.\\.|<=|>=|==|~=|.)"),
    reId("^[A-Za-z_][A-Za-z0-9_]*"),
    bi(3)
{
  // basic indent
  QSettings s;
  s.beginGroup("luaMode");
  bi = s.value("basicIndent", 3).toInt();
  // tokens
  for (int i=0; skeywords[i].text; i++)
    keywords[QString(skeywords[i].text)] = skeywords[i].type;
}


void 
QLuaModeLua::parseBlock(int pos, const QTextBlock &block, 
                        const QLuaModeUserData *idata, 
                        QLuaModeUserData *&odata )
{
  int len = block.length();
  QString text = block.text();
  UserData *data = new UserData;
  // input state
  if (idata)
    *data = *static_cast<const UserData*>(idata);
  // hack for statements that seem complete
  if (data->nodes.type() == Statement)
    setIndent(data->lastPos, data->nodes.next().indent());
  // process line
  gotLine(data, pos, block.length(), block.text());
  // flush parser stack on last block
  if (! block.next().isValid())
    gotToken(data, data->lastPos+1, 0, QString(), Eof);
  // output state
  odata = data;
}
  

// ========================================
// QLUAMODELUA - LEXICAL ANALYSIS


void
QLuaModeLua::gotLine(UserData *d, int pos, int len, QString s)
{
  // default indent
  if (pos == 0)
    setIndent(-1, 0);
  // lexical analysis
  int p = 0;
  int n = d->lexN;
  int r = d->lexPos - pos;
  int state = d->lexState;
  int slen = s.size();
  while (p < len)
    {
      int c = (p < slen) ? s[p].toAscii() : '\n';
      switch(state)
        {
        case 0:
          state = -1;
          if (c == '#') { 
            r = p; n = 0; state = -4; 
          }
          continue; 
        default:
        case -1:
          if (isspace(c)) {
            break;
          } if (isalpha(c) || c=='_') {
            r = p; state = -2; 
          } else if (c=='\'') {
            setIndentOverlay(pos+p+1, -1);
            r = p; n = -c; state = -3; 
          } else if (c=='\"') {
            setIndentOverlay(pos+p+1, -1);
            r = p; n = -c; state = -3;
          } else if (c=='[') {
            r = p; n = 0; state = -3;
            int t = p + 1;
            while (t < slen && s[t] == '=')
              t += 1;
            if (t < slen && s[t] == '[') {
              n = t - p;
              setIndentOverlay(pos+p, -1);
            } else {
              state = -1;
              gotToken(d, pos+p, 1, QString(), LBracket);
            }
          } else if (c=='-' && p+1 < slen && s[p+1]=='-') {
            r = p; n = 0; state = -4; 
            if (p+2 < slen && s[p+2]=='[') {
              int  t = p + 3;
              while (t < slen && s[t] == '=')
                t += 1;
              if (t < slen && s[t] == '[') {
                n = t - p - 2;
                setIndentOverlay(pos+p, 0);
                setIndentOverlay(pos+t+1, (n > 1) ? -1 :
                                 e->indentAfter(pos+t+1, +1) );
              }
            }
          } else if (reNum.indexIn(s,p,QRegExp::CaretAtOffset)>=0) {
            int l = reNum.matchedLength();
            QString m = s.mid(p,l);
            setFormat(pos+p, l, "number");
            gotToken(d, pos+p, l, m, Number);
            p += l - 1;
          } else if (reSym.indexIn(s,p,QRegExp::CaretAtOffset)>=0) {
            int l = reSym.matchedLength();
            QString m = s.mid(p,l);
            if (keywords.contains(m)) 
              gotToken(d, pos+p, l, QString(), keywords[m]);
            else
              gotToken(d, pos+p, l, m, Other);
            p += l - 1;
          }
          break;
        case -2: // identifier
          if (!isalnum(c) && c!='_') {
            QString m = s.mid(r, p-r);
            if (keywords.contains(m)) {
              setFormat(pos+r, p-r, "keyword");
              gotToken(d, pos+r, p-r, QString(), keywords[m]);
            } else 
              gotToken(d, pos+r, p-r, m, Identifier);
            state = -1; continue;
          }
          break;
        case -3: // string
          if (n <= 0 && (c == -n || c == '\n' || c == '\r')) {
            setFormat(pos+r,p-r+1,"string");
            setIndentOverlay(pos+p+1);
            gotToken(d, pos+r,p-r+1, QString(), String);
            state = -1;
          } else if (n <= 0 && c=='\\') {
            p += 1;
          } else if (n > 0 && c==']' && p>=n && s[p-n]==']') {
            int t = p - n + 1;
            while (t < slen && s[t] == '=')
              t += 1;
            if (t == p) {
              setFormat(pos+r,p-r+1,"string");
              setIndentOverlay(pos+p+1);
              gotToken(d, pos+r,p-r+1,QString(),String);
              state = -1;
            }
          }
          break;
        case -4: // comment
          if (n <= 0 && (c == '\n' || c == '\r')) {
            setFormat(pos+r, p-r, "comment");
            state = -1;
          } else if (n > 0 && c==']' && p>=n && s[p-n]==']') {
            int t = p - n + 1;
            while (t < slen && s[t] == '=')
              t += 1;
            if (t == p) {
              setFormat(pos+r, p-r+1, "comment");
              setIndentOverlay(pos+p-n, 2);
              setIndentOverlay(pos+p+1);
              state = -1;
            }
          }
          break;
        }
      p += 1;
    }
  // save state
  d->lexN = n;
  d->lexPos = r + pos;
  d->lexState = state;
  // format incomplete tokens
  if  (state == -4)
    setFormat(qMax(pos,pos+r),qMin(len,len-r),"comment");
  else if (state == -3)
    setFormat(qMax(pos,pos+r),qMin(len,len-r),"string");
}





// ========================================
// QLUAMODELUA - PARSING



#if DEBUG
QDebug operator<<(QDebug d, const TokenType &t)
{
  d.nospace();
# define DO(x) if (t==x) d << #x; else
  DO(Other) DO(Identifier) DO(Number) DO(String)
  DO(SemiColon) DO(ThreeDots) DO(Comma) DO(Dot) DO(Colon)
  DO(LParen) DO(RParen) DO(LBracket) 
  DO(RBracket) DO(LBrace) DO(RBrace) 
  DO(Kand) DO(Kfalse) DO(Kfunction) DO(Knil) 
  DO(Knot) DO(Kor) DO(Ktrue) DO(Kin)
  DO(Kbreak) DO(Kdo) DO(Kelse) DO(Kelseif) DO(Kend) DO(Kfor)
  DO(Kif) DO(Klocal) DO(Krepeat) DO(Kreturn)
  DO(Kthen) DO(Kuntil) DO(Kwhile)
  DO(Statement) DO(StatementCont) DO(Chunk)
  DO(FunctionBody) DO(FunctionName) DO(Eof)
# undef DO
  d << "<Unknown>";
  return d.space();
}
#endif



void
QLuaModeLua::gotToken(UserData *d, int pos, int len, 
                      QString s, TokenType ltype)
{
  PNode &n = d->nodes;
  TokenType ntype = n.type();
#if DEBUG
  qDebug() << " node:" << n << ntype << n.pos() << n.len() 
           << n.indent() << n.next().type() << n.next().next().type();
  if (s.isEmpty())
    qDebug() << "  token:" << pos << len << ltype;
  else
    qDebug() << "  token:" << pos << len << ltype << s;
#endif
  // close statements
  if ( ((ntype==Statement)
        && (ltype==Identifier || ltype==Kfunction) ) ||
       ((ntype==Statement || ntype==StatementCont || 
         ntype==Klocal || ntype==Kreturn ) 
        && (ltype==SemiColon || ltype>=FirstStrictKeyword) ) )
    {
      int epos = (ltype==SemiColon) ? pos+len : d->lastPos;
      int spos = n.pos();
      n = n.next();
      setBalance(spos, epos, n.type()==Chunk);
      setIndent(epos, n.indent());
    }
  if ((ntype == FunctionName || ntype == Kfunction) &&
      (ltype!=Identifier && ltype!=Dot && ltype!=Colon) )
    {
      if (ntype == FunctionName) n=n.next();
      ntype = n->type = FunctionBody;
      setIndent(pos, n.indent());
    }
  ntype = n.type();
  // fixup hacked indents
  if (ntype == StatementCont)
    n->type = Statement;
  if (d->lastPos < pos && ntype == Statement)
    setIndent(pos, n.indent());
  d->lastPos = pos + len;
  // parse
  switch (ltype)
    {
    badOne:
      {
        setIndent(pos, -1);
        while (n.type() != Chunk && n.len() == 0)  n = n.next();
        setErrorMatch(pos, len, n.pos(), n.len());
        n = n.next();
        setIndent(pos+len, n.indent());
        break;
      }

    case RParen:
      if (ntype != LParen)
        goto badOne;
      goto rightOne;

    case RBracket:
      if (ntype != LBracket) 
        goto badOne;
      goto rightOne;

    case RBrace:
      if (ntype != LBrace) 
        goto badOne;
      goto rightOne;

    case Kend:
      if (ntype!=Kdo && ntype!=Kelse && ntype!=Kthen
          && ntype!=Kfunction && ntype!=FunctionBody)
        goto badOne;
    rightOne:
      {
        setRightMatch(pos, len, n.pos(), n.len());
        int fpos = followMatch(n.pos(),n.len());
        int indent = n.indent();
        n = n.next();
        if (ltype < FirstKeyword)
          indent = qMin(qMax(0,indent-bi),e->indentAt(fpos));
        else
          indent = n.indent(); 
        setIndent(pos, indent);
        setIndent(pos+len, n.indent());
        setBalance(fpos, pos+len, n.type()==Chunk);
        break;
      }

    case Kuntil:
      if (ntype != Krepeat)
        goto badOne;
      {
        setRightMatch(pos, len, n.pos(), n.len());
        setIndent(pos, n.next().indent());
        setIndent(pos+len, n.indent());
        n->len = 0;
        n->type = StatementCont;
        break;
      }

    case Kthen:
      if (ntype!=Kif && ntype!=Kelseif)
        goto badOne;
      goto middleOne;

    case Kelse: case Kelseif:
      if (ntype!=Kthen)
        goto badOne;
    middleOne:
      {
        setMiddleMatch(pos, len, n.pos(), n.len());
        int fpos = followMatch(n.pos(), n.len());
        setIndent(pos, n.next().indent());
        setIndent(pos+len, n.indent());
        n->type = ltype;
        n->pos = pos;
        n->len = len;
        break;
      }

    case Kdo: 
      if (ntype==Kfor || ntype==Kwhile)
        goto middleOne;
      goto leftOne;

    case Kfunction:
      if (ntype == Klocal)
        goto middleOne;
      goto leftOne;

    case Kfor: case Kif: case Kwhile: 
    case Krepeat: case Klocal: case Kreturn:
    case LParen: case LBracket: case LBrace:
    leftOne:
      {
        int indent = n.indent() + bi;
        if (ltype == LBrace && ntype == StatementCont)
          indent = n.indent(); 
        else if (ltype < FirstKeyword)
          indent = e->indentAfter(pos+len);
        setIndent(pos, n.indent());
        n = PNode(ltype, pos, len, indent, n);
        setIndent(pos+len, indent);
        setLeftMatch(pos, len);
        break;
      }

    case SemiColon: 
    case Eof:
      break;

    case Identifier:
      if (ntype == Kfunction)
        n = PNode(FunctionName, pos, len, n.indent(), n);
      if (n.type() == FunctionName)
        setFormat(pos, len, "function");
      goto openStatement;

    case Dot: case Colon:
      if  (ntype == FunctionName)
        setFormat(pos, len, "function");
    case Kand: case Kor: case Knot: 
    case Kin: case Comma: case Other:
      if (n.type() == Statement)
        {
          n->type = StatementCont;
          setIndent(pos, n.indent());
        }
    default:
    openStatement:
      {
        if (ntype==Chunk || ntype==Kdo || ntype==Kthen || 
            ntype==Kelse || ntype==Krepeat || ntype==FunctionBody)
          {
            int indent = n.indent() + bi;
            n = PNode(Statement, pos, 0, indent, n);
            setIndent(pos+len, indent);
          }
        else if (ntype==Klocal)
          n->type = StatementCont;
        else if (ntype==Kreturn)
          n->type = Statement;        
        break;
      }
    }
}



// ========================================
// COMPLETION



static int 
comp_lex(QString s, int len, int state, int n, int &q)
{
  QChar z;
  int p = 0;
  while (p < len)
    {
      switch(state)
        {
        default:
        case -1: // misc
          if (isalpha(s[p].toAscii()) || s[p]=='_') {
            q = p; state = -2; 
          } else if (s[p]=='\'') {
            q = p+1; z = s[p]; n = 0; state = -3; 
          } else if (s[p]=='\"') {
            q = p+1; z = s[p]; n = 0; state = -3;
          } else if (s[p]=='[') {
            n = 0; state = -3;
            int t = p + 1;
            while (t < len && s[t] == '=')
              t += 1;
            if (t < len && s[t] == '[') {
              q = t + 1;
              n = t - p;
            } else
              state = -1;
          } else if (s[p]=='-' && s[p+1]=='-') {
            n = 0; state = -4; 
            if (s[p+2]=='[') {
              int t = p + 3;
              while (t < len && s[t] == '=')
                t += 1;
              if (t < len && s[t] == '[')
                n = t - p - 2;
            }
          }
          break;
        case -2: // identifier
          if (!isalnum(s[p].toAscii()) && s[p]!='_' && s[p]!='.' && s[p]!=':') {
            state = -1; continue;
          }
          break;
        case -3: // string
          if (n == 0 && s[p] == z) {
            state = -1;
          } else if (n == 0 && s[p]=='\\') {
            p += 1;
          } else if (n && s[p]==']' && p>=n && s[p-n]==']') {
            int t = p - n + 1;
            while (t < len && s[t] == '=')
              t += 1;
            if (t == p)
              state = -1;
          }
          break;
        case -4: // comment
          if (n == 0 && (s[p] == '\n' || s[p] == '\r')) {
            state = -1;
          } else if (n && s[p]==']' && p>=n && s[p-n]==']') {
            int t = p - n + 1;
            while (t < len && s[t] == '=')
              t += 1;
            if (t == p)
              state = -1;
          }
          break;
        }
      p += 1;
    }
  return state;
}


bool 
QLuaModeLua::doComplete()
{
  QString stem;
  QStringList completions;
  QTextCursor c = e->textCursor();
  QTextBlock b = c.block();
  int len = c.position() - b.position();
  QString text = b.text().left(len);
  int state = -1;
  int q = 0;
  int n = 0;
  QTextBlock pb = b.previous();
  if (pb.isValid())
    {
      UserData *data = static_cast<UserData*>(pb.userData());
      if (! data) 
        return false;
      state = data->lexState;
      n = data->lexN;
    }
  state = comp_lex(text, len, state, n, q);
  if (state == -3 && q >= 0 && q <= len)
    completions = computeFileCompletions(text.mid(q, len-q), n>0, stem);
  if (state == -2 && q >= 0 && q <= len)
    completions = computeSymbolCompletions(text.mid(q, len-q), stem);
  int selected = 0;
  if (completions.size() > 1)
    {
      qSort(completions.begin(), completions.end());
      for (int i=completions.size()-2; i>=0; i--)
        if (completions[i] == completions[i+1])
          completions.removeAt(i);
      selected = askCompletion(stem, completions);
    }
  if (selected >= 0 && selected < completions.size())
    {
      c.insertText(completions[selected]);
      e->setTextCursor(c);
      return true;
    }
  return false;
}


static const char *escape1 = "abfnrtv";
static const char *escape2 = "\a\b\f\n\r\t\v";


static QByteArray
unescapeString(const char *s)
{
  int c;
  QByteArray r;
  while ((c = *s++))
    {
      if (c != '\\')
        r += c;
      else {
        c = *s++;
        const char *e = strchr(escape1, c);
        if (e)
          r += escape2[e - escape1];
        else if (c >= '0' && c <= '7') {
          c = c - '0';
          if (*s >= '0' && *s <= '7')
            c = c * 8 + *s++ - '0';
          if (*s >= '0' && *s <= '7')
            c = c * 8 + *s++ - '0';
          r += c;
        } else 
          r += c;
      }
    }
  return r;
}


static QString
unescapeString(QString s)
{
  return QString::fromLocal8Bit(unescapeString(s.toLocal8Bit().constData()));
}


static QByteArray
escapeString(const char *s)
{
  int c;
  QByteArray r;
  while ((c = *s++))
    {
      const char *e;
      if (! isascii(c))
        r += c;
      else if (iscntrl(c) && (e = strchr(escape2, c)))
        r += escape1[e - escape2];
      else if (isprint(c) || isspace(c))
        r += c;
      else {
        char buffer[8];
        sprintf(buffer, "\\%03o", c);
        r += buffer;
      }
    }
  return r;
}


static QString
escapeString(QString s)
{
  return QString::fromLocal8Bit(escapeString(s.toLocal8Bit().constData()));
}


QStringList 
QLuaModeLua::computeFileCompletions(QString s, bool escape, QString &stem)
{
  QStringList list;
  s.remove(QRegExp("^.*\\s"));
  stem = s;
  if (escape)
    stem = unescapeString(s);
  fileCompletion(stem, list);
  if (escape) 
    {
      QStringList nl;
      foreach(QString s, list)
        nl += escapeString(s);
      stem = escapeString(stem);
      list = nl;
    }
  return list;
}


static const char *
comp_keywords[] = 
  {
    "and", "break", "do", "else", "elseif", 
    "end", "false", "for", "function",
    "if", "in", "local", "nil", "not", 
    "or", "repeat", "return", "then",
    "true", "until", "while", 0
  };


QStringList 
QLuaModeLua::computeSymbolCompletions(QString s, QString &stem)
{
  QStringList list;
  QByteArray f = s.toLocal8Bit();
  int flen = f.size();
  // stem
  stem = s.remove(QRegExp("^.*[.:]"));
  // keywords
  for (const char **k = comp_keywords; *k; k++)
    if (!strncmp(f.constData(), *k, flen))
      list += QString::fromLocal8Bit(*k + flen);
  // symbols
  QtLuaEngine *engine = QLuaApplication::engine();
  if (engine)
    {
      QtLuaLocker lua(engine, 250);
      struct lua_State *L = lua;
      if (lua)
        {
          lua_pushcfunction(L, luaQ_complete);
          lua_pushlstring(L, f.constData(), flen);
          if (!lua_pcall(L, 1, 1, 0) && lua_istable(L, -1)) {
            int n = lua_objlen(L, -1);
            for (int j=1; j<=n; j++) {
              lua_rawgeti(L, -1, j);
              list += QString::fromLocal8Bit(lua_tostring(L, -1));
              lua_pop(L, 1);
            }
          }
          lua_pop(L, 1);
        }
      else
        {
          QWidget *w = e->window();
          QLuaMainWindow *m = qobject_cast<QLuaMainWindow*>(w);
          if (m)
            m->showStatusMessage(tr("Auto-completions is restricted "
                                    "while Lua is running.") );
          QLuaApplication::beep();
        }
    }
  return list;
}







// ========================================
// FACTORY


static QLuaModeFactory<QLuaModeLua> textModeFactory("Lua", "lua");



// ========================================
// MOC


#include "qluamode_lua.moc"





/* -------------------------------------------------------------
   Local Variables:
   c++-font-lock-extra-types: ("\\sw+_t" "\\(lua_\\)?[A-Z]\\sw*[a-z]\\sw*")
   End:
   ------------------------------------------------------------- */
