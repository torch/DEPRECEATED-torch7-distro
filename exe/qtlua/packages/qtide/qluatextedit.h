// -*- C++ -*-

#ifndef QLUATEXTEDIT_H
#define QLUATEXTEDIT_H

#include "qtide.h"


#include <QDialog>
#include <QFile>
#include <QList>
#include <QObject>
#include <QPlainTextEdit>
#include <QSize>
#include <QString>
#include <QSyntaxHighlighter>
#include <QTextBlock>
#include <QTextCharFormat>
#include <QTextCursor>
#include <QTextDocument>
#include <QTextEdit>
#include <QVariant>
#include <QWidget>

class QLuaEditor;
class QLuaTextEdit;
class QLuaTextEditMode;
class QLuaTextEditModeFactory;


// Text editor widget


class QTIDE_API QLuaTextEdit : public QPlainTextEdit
{
  Q_OBJECT
  Q_PROPERTY(bool showLineNumbers READ showLineNumbers WRITE setShowLineNumbers)
  Q_PROPERTY(bool autoComplete READ autoComplete WRITE setAutoComplete)
  Q_PROPERTY(bool autoIndent READ autoIndent WRITE setAutoIndent)
  Q_PROPERTY(bool autoHighlight READ autoHighlight WRITE setAutoHighlight)
  Q_PROPERTY(bool autoMatch READ autoMatch WRITE setAutoMatch)
  Q_PROPERTY(bool tabExpand READ tabExpand WRITE setTabExpand)
  Q_PROPERTY(int tabSize READ tabSize WRITE setTabSize)
  Q_PROPERTY(QSize sizeInChars READ sizeInChars WRITE setSizeInChars)

public:

  QLuaTextEdit(QWidget *parent=0);
  bool showLineNumbers() const;
  bool autoComplete() const;
  bool autoIndent() const;
  bool autoHighlight() const;
  bool autoMatch() const;
  bool tabExpand() const;
  int tabSize() const;
  QSize sizeInChars() const;
  
  int indentAt(int pos);
  int indentAt(int pos, QTextBlock block);
  int indentAfter(int pos, int dpos=0);
  int getBlockIndent(QTextBlock block);
  int getBlockIndent(QTextBlock block, int &indent);
  int setBlockIndent(QTextBlock block, int indent);
  bool readFile(QFile &file);
  bool writeFile(QFile &file);
  bool print(QPrinter *printer);
  
  Q_INVOKABLE QLuaTextEditMode *editorMode() const;
  Q_INVOKABLE virtual QDialog *makeFindDialog();
  Q_INVOKABLE virtual QDialog *makeReplaceDialog();
  Q_INVOKABLE virtual QDialog *makeGotoDialog();
  Q_INVOKABLE virtual void prepareDialog(QDialog *dialog);
  
  static QTextCharFormat format(QString key);
  static void setFormat(QString key, QTextCharFormat format);
  QRectF blockBoundingGeometry(const QTextBlock &block) const;
  
public slots:
  void setShowLineNumbers(bool b);
  void setAutoComplete(bool b);
  void setAutoIndent(bool b);
  void setAutoHighlight(bool b);
  void setAutoMatch(bool b);
  void setTabExpand(bool b);
  void setTabSize(int s);
  void setSizeInChars(QSize size);
  bool setEditorMode(QLuaTextEditModeFactory *modeFactory = 0); 
  bool setEditorMode(QString suffix);
  bool readFile(QString fname);
  bool writeFile(QString fname);
  void showLine(int lineno);
  void reHighlight();

signals:
  void settingsChanged();

protected:
  virtual QSize sizeHint() const;
  virtual void keyPressEvent(QKeyEvent *event);
  
public:
  class Private;
  class GotoDialog;
  class FindDialog;
  class ReplaceDialog;
  class LineNumbers;
private:
  friend class QLuaSdiMain;
  Private *d;
};




// Text editor language support 


class QTIDE_API QLuaTextEditModeFactory
{
public:
  virtual ~QLuaTextEditModeFactory();
  QLuaTextEditModeFactory(const char *name, const char *suffixes);
  virtual QLuaTextEditMode *create(QLuaTextEdit *parent) = 0;
  virtual QString name();
  virtual QString filter();
  virtual QStringList suffixes();
  static QList<QLuaTextEditModeFactory*> factories();
private:
  const char *const name_;
  const char *const suffixes_;
  QLuaTextEditModeFactory *next;
  QLuaTextEditModeFactory *prev;
  static QLuaTextEditModeFactory *first;
  static QLuaTextEditModeFactory *last;
};


class QTIDE_API QLuaTextEditMode : public QObject
{
  Q_OBJECT
protected:
  QLuaTextEdit * const e;
  QLuaTextEditModeFactory * const f;
public:
  QLuaTextEditMode(QLuaTextEditModeFactory *f, QLuaTextEdit *e);
  QLuaTextEditModeFactory *factory() const { return f; }
  virtual bool supportsHighlight() { return false; }
  virtual bool supportsMatch()     { return false; }
  virtual bool supportsBalance()   { return false; }
  virtual bool supportsIndent()    { return false; }
  virtual bool supportsComplete()  { return false; }
  virtual bool supportsLua()       { return false; }
  virtual QString name()           { return f->name(); }
  virtual QString filter()         { return f->filter(); }
  virtual QStringList suffixes()   { return f->suffixes(); }
  virtual QSyntaxHighlighter *highlighter() { return 0; }
public slots:
  virtual bool doEnter()             { return false; }
  virtual bool doTab()               { return false; }
  virtual bool doMatch()             { return false; }
  virtual bool doBalance()           { return false; }
  virtual bool doComplete()          { return false; }
};



#endif


/* -------------------------------------------------------------
   Local Variables:
   c++-font-lock-extra-types: ("\\sw+_t" "\\(lua_\\)?[A-Z]\\sw*[a-z]\\sw*")
   End:
   ------------------------------------------------------------- */

