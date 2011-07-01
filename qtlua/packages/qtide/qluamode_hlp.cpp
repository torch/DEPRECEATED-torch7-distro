/* -*- C++ -*- */

#include <QtGlobal>
#include <QDebug>
#include <QList>
#include <QRegExp>
#include <QStack>


#include "qluatextedit.h"
#include "qluamode.h"

#include <string.h>



// ========================================
// USERDATA


namespace {

  struct UserData : public QLuaModeUserData
  {
    bool verbatim;
    UserData() : verbatim(false) { }
    virtual int highlightState() { return (verbatim) ? 1 : 0; }
  };
  
}




// ========================================
// QLUAMODETEXT




class QLuaModeHelp : public QLuaMode
{
  Q_OBJECT
public:
  QLuaModeHelp(QLuaTextEditModeFactory *f, QLuaTextEdit *e);
  virtual bool doEnter();
  virtual bool supportsHighlight() { return true; }
  virtual bool supportsMatch() { return true; }
  virtual bool supportsBalance() { return false; }
  virtual bool supportsIndent() { return true; }
  virtual void parseBlock(int pos, const QTextBlock &block, 
                          const QLuaModeUserData *idata, 
                          QLuaModeUserData *&odata );
  void gotLine(UserData *d, int pos, int len, QString);
private:
  QRegExp reSection, reHRule, reIndent, reEmpty, reToken;
  QRegExp reFormat1, reFormat2, reEVerb;
};


QLuaModeHelp::QLuaModeHelp(QLuaTextEditModeFactory *f, QLuaTextEdit *e)
  : QLuaMode(f,e), 
    reSection("^\\-\\-\\-\\+.*$"),
    reHRule("^(\\-\\-\\-)+"),
    reIndent("^(   |\t)+(\\*|[1aAiI]\\.|\\$(?=.+:\\s+.*\\$))[ \t]+"),
    reEmpty("^\\s*$"),
    reToken("(<[^>]*>?|#\\w+|\\[\\[|\\]\\[|\\]\\]|==|__|\\*|=|_)"),
    reFormat1("^(==|__|\\*|=|_)(\\S+\\S?)(\\1)"),
    reFormat2("^(==|__|\\*|=|_)(\\S.*\\S)(\\1)"),
    reEVerb("</verbatim>")
{
  reFormat1.setMinimal(false);
  reFormat2.setMinimal(true);
}


bool
QLuaModeHelp::doEnter()
{
  e->textCursor().insertBlock();
  return true;
}


void 
QLuaModeHelp::parseBlock(int pos, const QTextBlock &block, 
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
  // output state
  odata = data;
}


void
QLuaModeHelp::gotLine(UserData *d, int pos, int len, QString line)
{
  int i = 0;
  int matchPos = 0;
  int matchLen = 0;
  QString matchToken;
  while (i < len)
    {
      int p;
      if (d->verbatim)
        {
          if ((p = reEVerb.indexIn(line, i)) >= 0)
            {
              int l = reEVerb.matchedLength();
              setFormat(pos, p, "string");
              setIndentOverlay(pos+p);
              d->verbatim = false;
              i = p;
            }
          else
            {
              i = len;
              setFormat(pos, len-1, "string");
            }
          continue;
        }
      if (i == 0)
        {
          if ((pos == 0) || (p = reSection.indexIn(line, i)) >= 0)
            {
              setFormat(pos,len-1,"comment");
              setIndent(pos+0, 0);
              i = len;
              continue;
            }
          if ((p = reHRule.indexIn(line, i)) >= 0)
            {
              int l = reHRule.matchedLength();
              setFormat(pos, l, "keyword");
              setIndent(pos, -1);
              i = i + l;
              continue;
             
            }
          if ((p = reIndent.indexIn(line, i)) >= 0)
            {
              int l = reIndent.matchedLength();
              int m = reIndent.pos(2);
              setFormat(pos, l, "keyword");
              setIndent(pos, e->indentAt(m));
              setIndent(pos+l, e->indentAt(pos+l));
              i = i + l;
              continue;
            }
          if ((p = reEmpty.indexIn(line, i)) >= 0)
            {
              setIndent(pos+1, -1);
              i = len;
              continue;
            }
        }
      if ((p = reToken.indexIn(line, i)) >= 0)
        {
          int l = reToken.matchedLength();
          QString k = line.mid(p, l);
          setFormat(pos+p, l, "keyword");
          if (k == "]]" && matchLen>0)
            {
              if (matchToken == "[[") {
                int opos = matchPos+matchLen;
                setFormat(opos, pos+p-opos, "url");
              }
              setRightMatch(pos+p, l, matchPos, matchLen);
              matchLen = 0;
            }
          else if (k == "][" && matchLen>0)
            {
              if (matchToken == "[[") {
                int opos = matchPos+matchLen;
                setFormat(opos, pos+p-opos, "url");
                setMiddleMatch(pos+p, l, matchPos, matchLen);
              } else
                setErrorMatch(pos+p, l, matchPos, matchLen);
              matchPos = pos+p;
              matchLen = l;
              matchToken = k;
            }
          else if (k == "[[" && matchLen<=0)
            {
              setLeftMatch(pos+p, l);
              matchPos = pos+p;
              matchLen = l;
              matchToken = k;
            }
          else if (k == "[[" || k == "][" || k == "]]")
            {
              setErrorMatch(pos+p, l, matchPos, matchLen);
              matchLen = 0;
            }
          else if (k[0] ==  '#')
            {
              setFormat(pos+p, l, "url");
            }
          else if (k == "<verbatim>")
            {
              d->verbatim = true;
              setFormat(pos+p+l, len-p-1, "string");
              setIndentOverlay(pos+p+l, -1);
              i = len;
              continue;
            }
          else if (k[0] != '<')
            {
              int q;
              int q1 = p;
              int q2 = p;
              if ((q = reFormat1.indexIn(line, p, 
                                         QRegExp::CaretAtOffset) ) == p)
                {
                  l = reFormat1.matchedLength();
                  k = reFormat1.cap(1);
                  q1 = reFormat1.pos(2);
                  q2 = reFormat1.pos(3);
                }
              else if ((q = reFormat2.indexIn(line, p, 
                                              QRegExp::CaretAtOffset) ) == p)
                {
                  l = reFormat2.matchedLength();
                  k = reFormat2.cap(1);
                  q1 = reFormat2.pos(2);
                  q2 = reFormat2.pos(3);
                }
              QTextCharFormat fmt = e->format("Help/string");
              if (k == "__" || k == "==" || k == "*")
                fmt.setFontWeight(QFont::Bold);
              setFormat(pos+p, l, (q>=0) ? "keyword" : "normal");
              if (q >= 0) 
                {
                  setLeftMatch(pos+p, q1-p);
                  setFormat(pos+q1, q2-q1, fmt);
                  setRightMatch(pos+q2, p+l-q2, pos+p, q1-p);
                }
            }
          i = p+l;
          continue;
        }
      i = len;
    }
}


// ========================================
// FACTORY


const char *helpName = QT_TRANSLATE_NOOP("QLuaTextEditMode", "Help");

static QLuaModeFactory<QLuaModeHelp> textModeFactory(helpName, "hlp");



// ========================================
// MOC


#include "qluamode_hlp.moc"





/* -------------------------------------------------------------
   Local Variables:
   c++-font-lock-extra-types: ("\\sw+_t" "\\(lua_\\)?[A-Z]\\sw*[a-z]\\sw*")
   End:
   ------------------------------------------------------------- */
