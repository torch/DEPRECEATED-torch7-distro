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

  enum LexState {
    LexTypeMask = 0xfff,
    LexStart = 0,
    LexDefault,
    LexString,
    LexChar,
    LexComment,
    
    LexCppMask = 0xfff00,
    LexCppStart = 0x1000,
    LexCpp = 0x2000,
  };

  enum TokenType {
    // generic tokens
    Other, Identifier, Type, Number, String,
    // tokens
    SemiColon, Colon, DoubleColon,
    // type keywords
    Kbool, Kchar, Kdouble, Kfloat, Kint,
    Klong, Kshort, Kvoid, Kwchar_t,
    // type annunciators
    Ksigned, Kconst, Kunsigned, Kvolatile, Kauto,
    Kexplicit, Kmutable, Kextern, Kstatic, Kregister,
    Kinline, Ktemplate, Kvirtual,
    Kclass, Kenum, Knamespace, Ktypedef,
    Kfriend, Kstruct, Kunion,
    // expression keywords
    Kconst_cast, Kdelete, Kdynamic_cast, Kfalse,
    Knew, Koperator, Kstatic_cast, Kreinterpret_cast,
    Kreturn, Ksizeof, Ktrue, Kthis, Kthrow,
    Ktypeid, Ktypename, Kusing,
    // flow keywords
    Kasm, Kbreak, Kcase, Kcatch, Kcontinue, Kdefault,
    Kdo, Kelse, Kexport, Kfor, Kgoto, Kif,
    Kswitch, Ktry, Kwhile,
    // scope keywords
    Kprivate, Kprotected, Kpublic,
    // special
    LParen, RParen, LBrace, RBrace, LBracket, RBracket,
    Expr, Block, Main, Eof,
    FirstTypeKeyword = Kbool,
    LastTypeKeyword = Kwchar_t,
    FirstTypeAnnunciator = Ksigned,
    LastTypeAnnunciator = Kunion
  };

  struct Keywords { 
    const char *text; 
    TokenType type; 
  };
  
  Keywords skeywords[] = {
    {"bool", Kbool}, {"char", Kchar}, {"double", Kdouble}, 
    {"float", Kfloat}, {"int", Kint},
    {"long", Klong}, {"short", Kshort}, {"void", Kvoid}, 
    {"wchar_t", Kwchar_t},
    {"signed", Ksigned}, {"const", Kconst}, 
    {"unsigned", Kunsigned}, {"volatile", Kvolatile}, {"auto", Kauto},
    {"explicit", Kexplicit}, {"mutable", Kmutable}, 
    {"extern", Kextern}, {"static", Kstatic}, {"register", Kregister},
    {"inline", Kinline}, {"template", Ktemplate}, {"virtual", Kvirtual},
    {"class", Kclass}, {"enum", Kenum}, {"namespace", Knamespace}, 
    {"typedef", Ktypedef}, {"friend", Kfriend}, {"struct", Kstruct}, 
    {"union", Kunion}, {"const_cast", Kconst_cast}, {"delete", Kdelete}, 
    {"dynamic_cast", Kdynamic_cast}, {"false", Kfalse}, {"new", Knew}, 
    {"operator", Koperator}, {"static_cast", Kstatic_cast}, 
    {"reinterpret_cast", Kreinterpret_cast},
    {"return", Kreturn}, {"sizeof", Ksizeof}, {"true", Ktrue}, 
    {"this", Kthis}, {"throw", Kthrow},
    {"typeid", Ktypeid}, {"typename", Ktypename}, {"using", Kusing},
    {"asm", Kasm}, {"break", Kbreak}, {"case", Kcase}, 
    {"catch", Kcatch}, {"continue", Kcontinue}, {"default", Kdefault},
    {"do", Kdo}, {"else", Kelse}, {"export", Kexport}, 
    {"for", Kfor}, {"goto", Kgoto}, {"if", Kif},
    {"switch", Kswitch}, {"try", Ktry}, {"while", Kwhile},
    {"private", Kprivate}, {"protected", Kprotected}, {"public", Kpublic},
    {";", SemiColon}, {":", Colon}, {"::", DoubleColon}, 
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
    return (n) ? n->type : Main; 
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
    // parser state
    PNode nodes;
    int lastPos;
    // initialize
    UserData() : lexState(0), lexPos(0), lastPos(0) {}
    virtual int highlightState() { return lexState; }
  };

}




// ========================================
// QLUAMODELUA


class QLuaModeC : public QLuaMode
{
  Q_OBJECT
public:
  QLuaModeC(QLuaTextEditModeFactory *f, QLuaTextEdit *e);
  virtual bool supportsMatch() { return false; }
  virtual bool supportsBalance() { return false; }
  virtual bool supportsIndent() { return false; }
  void gotLine(UserData *d, int pos, int len, QString);
  void gotToken(UserData *d, int pos, int len, QString, TokenType);
  virtual void parseBlock(int pos, const QTextBlock &block, 
                          const QLuaModeUserData *idata, 
                          QLuaModeUserData *&odata );
private:
  QMap<QString,TokenType> keywords;
  QRegExp reNum, reId, reType, reSym; 
  int bi;
};


QLuaModeC::QLuaModeC(QLuaTextEditModeFactory *f, QLuaTextEdit *e)
  : QLuaMode(f,e), bi(2)
{
  // regexps
  reNum = QRegExp("^(0x[0-9a-fA-F]+|0b[01]+|0[0-7]+"
                  "|(\\.[0-9]+|[0-9]+(\\.[0-9]*)?)([Ee][-+]?[0-9]*)?)");
  reId = QRegExp("^[A-Za-z_$][A-Za-z0-9_$]*");
  reType = QRegExp("^(.+_t|(lua_)?[A-Z].*)$");
  reSym = QRegExp("^(::|.)");
  // basic indent
  QSettings s;
  s.beginGroup("CMode");
  bi = s.value("basicIndent", 2).toInt();
  // tokens
  for (int i=0; skeywords[i].text; i++)
    keywords[QString(skeywords[i].text)] = skeywords[i].type;
}


void 
QLuaModeC::parseBlock(int pos, const QTextBlock &block, 
                        const QLuaModeUserData *idata, 
                        QLuaModeUserData *&odata )
{
  int len = block.length();
  QString text = block.text();
  UserData *data = new UserData;
  // input state
  if (idata)
    *data = *static_cast<const UserData*>(idata);
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
QLuaModeC::gotLine(UserData *d, int pos, int len, QString s)
{
  // default indent
  if (pos == 0)
    setIndent(-1, 0);
  // lexical analysis
  int p = 0;
  int r = d->lexPos - pos;
  int state = d->lexState & LexTypeMask;
  int cppstate = d->lexState & LexCppMask;
  int slen = s.size();
  while (p < len)
    {
      int c = (p < slen) ? s[p].toAscii() : '\n';
      switch(state)
        {
        case LexStart:
          if (isspace(c)) break;
          state = LexDefault;
          if (c == '#') {
            cppstate = LexCppStart;
            setIndentOverlay(pos+p+1, e->indentAfter(pos+p+1,+2));
            p += 1;
          }
          continue; 
        default:
        case LexDefault:
          if ((cppstate) && (c == '\\') && (p+1 == slen)) {
            p = len; continue;
          } else if (isspace(c)) {
            if (c == '\n' || c == '\r') {
              cppstate = 0; state = LexStart;
              setIndentOverlay(pos+p);
            }
            break;
          } else if (c=='\'') {
            setIndentOverlay(pos+p+1, -1);
            r = p; state = LexChar; 
          } else if (c=='\"') {
            setIndentOverlay(pos+p+1, -1);
            r = p; state = LexString;
          } else if (c=='/' && p+1 < slen && s[p+1]=='/') {
            setFormat(pos+p, len-p, "comment");
            p = len; continue;
          } else if (c=='/' && p+1 < slen && s[p+1]=='*') {
            r = p; state = LexComment;
            setIndentOverlay(pos+p+1, -1);
          } else if ((isalpha(c) || c=='_') &&
                     (reId.indexIn(s,p,QRegExp::CaretAtOffset)>=0) ) {
            int l = reId.matchedLength();
            QString m = s.mid(p,l);
            TokenType type = Identifier;
            if (cppstate == LexCppStart) {
              setFormat(pos+p, l, "cpp");
            } else if (keywords.contains(m)) {
              type = keywords[m];
              if (type >= FirstTypeKeyword && type <= LastTypeKeyword)
                setFormat(pos+p, l, "type");
              else
                setFormat(pos+p, l, "keyword");
            } else if (m.contains(reType)) {
              type = Type;
              setFormat(pos+p, l, "type");
            }
            if (!cppstate)
              gotToken(d, pos+r, p-r, m, type);
            p += l - 1;
          } else if (reNum.indexIn(s,p,QRegExp::CaretAtOffset)>=0) {
            int l = reNum.matchedLength();
            QString m = s.mid(p,l);
            setFormat(pos+p, l, "number");
            if (! cppstate)
              gotToken(d, pos+p, l, m, Number);
            p += l - 1;
          } else if (reSym.indexIn(s,p,QRegExp::CaretAtOffset)>=0) {
            int l = reSym.matchedLength();
            QString m = s.mid(p,l);
            TokenType type = Other;
            if (keywords.contains(m)) 
              type = keywords[m];
            if (! cppstate)
              gotToken(d, pos+p, l, m, type);
            p += l - 1;
          }
          break;
        case LexString:
          if (c == '\"' || c == '\n' || c == '\r') { 
            setFormat(pos+r,p-r+1,"string");
            setIndentOverlay(pos+p+1);
            if (!cppstate)
              gotToken(d, pos+r,p-r+1, QString(), String);
            state = LexDefault;
          } else if (c=='\\')
            p += 1;
          break;
        case LexChar:
          if (c == '\'' || c == '\n' || c == '\r') { 
            setFormat(pos+r,p-r+1,"string");
            setIndentOverlay(pos+p+1);
            if (!cppstate)
              gotToken(d, pos+r,p-r+1, QString(), String);
            state = LexDefault;
          } else if (c=='\\')
            p += 1;
          break;
        case LexComment: // comment
          if (c=='*' && p+1 < slen && s[p+1]=='/') {
            setFormat(pos+r, p-r+2, "comment");
            setIndentOverlay(pos+p+1);
            state = LexDefault;
          }
          break;
        }
      if (cppstate == LexCppStart && !isspace(c))
        cppstate = LexCpp;
      p += 1;
    }
  // save state
  d->lexPos = r + pos;
  d->lexState = state | cppstate;
  // format incomplete tokens
  if  (state == LexComment)
    setFormat(qMax(pos,pos+r),qMin(len,len-r),"comment");
  else if (state == LexString || state == LexChar)
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
  DO(SemiColon) DO(Colon) DO(DoubleColon)
  DO(Kbool) DO(Kchar) DO(Kdouble) DO(Kfloat) DO(Kint)
  DO(Klong) DO(Kshort) DO(Kvoid) DO(Kwchar_t)
  DO(Ksigned) DO(Kconst) DO(Kunsigned) DO(Kvolatile) DO(Kauto)
  DO(Kexplicit) DO(Kmutable) DO(Kextern) DO(Kstatic) DO(Kregister)
  DO(Kinline) DO(Ktemplate) DO(Kvirtual)
  DO(Kclass) DO(Kenum) DO(Knamespace) DO(Ktypedef)
  DO(Kfriend) DO(Kstruct) DO(Kunion)
  DO(Kconst_cast) DO(Kdelete) DO(Kdynamic_cast) DO(Kfalse)
  DO(Knew) DO(Koperator) DO(Kstatic_cast) DO(Kreinterpret_cast)
  DO(Kreturn) DO(Ksizeof) DO(Ktrue) DO(Kthis) DO(Kthrow)
  DO(Ktypeid) DO(Ktypename) DO(Kusing)
  DO(Kasm) DO(Kbreak) DO(Kcase) DO(Kcatch) DO(Kcontinue) DO(Kdefault)
  DO(Kdo) DO(Kelse) DO(Kexport) DO(Kfor) DO(Kgoto) DO(Kif)
  DO(Kswitch) DO(Ktry) DO(Kwhile)
  DO(Kprivate) DO(Kprotected) DO(Kpublic)
  DO(Expr) DO(Block) DO(Eof)
# undef DO
  d << "<Unknown>";
  return d.space();
}
#endif



void
QLuaModeC::gotToken(UserData *d, int pos, int len, 
                      QString s, TokenType ltype)
{
#if DEBUG
  qDebug() << pos << len << ltype << s;
#endif
}






// ========================================
// FACTORY


static QLuaModeFactory<QLuaModeC> textModeFactory("C", 
                                                  "c;h;cpp;hpp;c++;h++;C;H");



// ========================================
// MOC


#include "qluamode_c.moc"





/* -------------------------------------------------------------
   Local Variables:
   c++-font-lock-extra-types: ("\\sw+_t" "\\(lua_\\)?[A-Z]\\sw*[a-z]\\sw*")
   End:
   ------------------------------------------------------------- */
