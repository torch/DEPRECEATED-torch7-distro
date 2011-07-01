// -*- C++ -*-

#ifndef QLUAMODE_H
#define QLUAMODE_H

#include "qtide.h"
#include "qluatextedit.h"

#include <QList>
#include <QObject>
#include <QSize>
#include <QString>
#include <QStringList>
#include <QSyntaxHighlighter>
#include <QTextBlock>
#include <QTextBlockUserData>
#include <QTextCharFormat>
#include <QTextCursor>
#include <QTextDocument>
#include <QTextEdit>
#include <QVariant>
#include <QWidget>



// Class to store user data for each block

class QTIDE_API QLuaModeUserData : public QTextBlockUserData
{
public:
  virtual int highlightState();
};



// A convenient base class for creating modes

class QTIDE_API QLuaMode : public QLuaTextEditMode
{
  Q_OBJECT
public:
  QLuaMode(QLuaTextEditModeFactory *f, QLuaTextEdit *e);
  virtual QSyntaxHighlighter *highlighter();
  virtual bool supportsHighlight() { return true; }
  virtual bool supportsMatch() { return true; }
  virtual bool supportsBalance() { return true; }
  virtual bool supportsIndent() { return true; }
protected:
  virtual void setFormat(int pos, int len, QString format);
  virtual void setFormat(int pos, int len, QTextCharFormat format);
  virtual void setLeftMatch(int pos, int len);
  virtual void setMiddleMatch(int pos, int len, int pp, int pl);
  virtual void setRightMatch(int pos, int len, int pp, int pl);
  virtual void setErrorMatch(int pos, int len, int pp, int pl, bool h=true);
  virtual  int followMatch(int pos, int len);
  virtual void setBalance(int fpos, int tpos);
  virtual void setBalance(int fpos, int tpos, bool outer);
  virtual void setIndent(int pos, int indent);
  virtual void setIndentOverlay(int pos, int indent=-2);
  virtual  int askCompletion(QString stem, QStringList comps);
  virtual void fileCompletion(QString &stem, QStringList &comps);
protected:
  virtual void parseBlock(int pos, const QTextBlock &block, 
                          const QLuaModeUserData *idata, 
                          QLuaModeUserData *&odata ) = 0;
public slots:
  virtual bool doTab();
  virtual bool doEnter();
  virtual bool doMatch();
  virtual bool doBalance();
public:
  class CompModel;
  class CompView;
  class Highlighter;
  class Private;
 private:
  Private *d;
};


// A template class for creating mode factories


template<class T>
class QTIDE_API QLuaModeFactory : public QLuaTextEditModeFactory
{
public:
  QLuaModeFactory(const char *n, const char *s) 
    : QLuaTextEditModeFactory(n,s) {}

  virtual QLuaTextEditMode *create(QLuaTextEdit *parent) 
    { return new T(this,parent); }
};


#endif




/* -------------------------------------------------------------
   Local Variables:
   c++-font-lock-extra-types: ("\\sw+_t" "\\(lua_\\)?[A-Z]\\sw*[a-z]\\sw*")
   End:
   ------------------------------------------------------------- */

