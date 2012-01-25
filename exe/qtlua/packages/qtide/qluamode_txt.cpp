/* -*- C++ -*- */

#include <QtGlobal>
#include <QDebug>
#include <QList>
#include <QRegExp>
#include <QSharedData>
#include <QSharedDataPointer>


#include "qluatextedit.h"
#include "qluamode.h"

#include <string.h>



// ========================================
// USERDATA


namespace {

  struct Match;
  
  typedef QSharedDataPointer<Match> PMatch;
  
  struct Match : public QSharedData
  {
    PMatch next;
    char type;
    int pos;
  };
  
  struct UserData : public QLuaModeUserData
  {
    PMatch stack;
  };
  
}




// ========================================
// QLUAMODETEXT




class QLuaModeText : public QLuaMode
{
  Q_OBJECT
public:
  QLuaModeText(QLuaTextEditModeFactory *f, QLuaTextEdit *e);
  virtual bool doEnter();
  virtual void parseBlock(int pos, const QTextBlock &block, 
                          const QLuaModeUserData *idata, 
                          QLuaModeUserData *&odata );
private:
  QRegExp reHighlight;
};


QLuaModeText::QLuaModeText(QLuaTextEditModeFactory *f, QLuaTextEdit *e)
  : QLuaMode(f,e), 
    reHighlight("^[|>]")
{
}


bool
QLuaModeText::doEnter()
{
  e->textCursor().insertBlock();
  return true;
}


void 
QLuaModeText::parseBlock(int pos, const QTextBlock &block, 
                         const QLuaModeUserData *idata, 
                         QLuaModeUserData *&odata )
{
  int len = block.length();
  QString text = block.text();
  UserData *data = new UserData;
  
  // input state
  if (idata)
    *data = *static_cast<const UserData*>(idata);
  
  // highlight
  if (text.contains(reHighlight))
    setFormat(pos, len, "quote");
  
  // indentation
  int indent;
  int cpos = e->getBlockIndent(block, indent);
  if (indent >= 0)
    setIndent(cpos+1, indent);
  else
    setIndent(block.position()+1, -1);
  
  // matches
  for (int i=0; i<len; i++)
    {
      char ic = text[i].toAscii();
      if (ic=='(' || ic=='[' || ic=='{')
        {
          PMatch m(new Match);
          m->type = ic;
          m->pos = pos + i;
          m->next = data->stack;
          data->stack = m;
          setLeftMatch(pos + i, 1);
        }
      else if (ic==')' || ic==']' || ic=='}')
        {
          PMatch m = data->stack;
          if (m)
            {
              if (m->type == strchr("()[]{}", ic)[-1])
                setRightMatch(pos + i, 1, m->pos, 1);
              else
                setErrorMatch(pos + i, 1, m->pos, 1);                
              setBalance(m->pos, pos+i+1, !m->next);
              data->stack = m->next;
            }
        }
    }

  // output state
  odata = data;
}
  






// ========================================
// FACTORY


const char *textName = QT_TRANSLATE_NOOP("QLuaTextEditMode", "Text");

static QLuaModeFactory<QLuaModeText> textModeFactory(textName, "txt");



// ========================================
// MOC


#include "qluamode_txt.moc"





/* -------------------------------------------------------------
   Local Variables:
   c++-font-lock-extra-types: ("\\sw+_t" "\\(lua_\\)?[A-Z]\\sw*[a-z]\\sw*")
   End:
   ------------------------------------------------------------- */
