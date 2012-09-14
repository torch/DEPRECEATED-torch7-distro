// -*- C++ -*-

#ifndef QLUABROWSER_H
#define QLUABROWSER_H

#include "qtide.h"
#include "qluamainwindow.h"

#include <QFile>
#include <QObject>
#include <QUrl>
#include <QWidget>
#if HAVE_QTWEBKIT
# include <QWebPage>
# include <QWebView>
# include <QWebFrame>
#else
class QWebPage;
class QWebView;
#endif

// Text Editor

class QTIDE_API QLuaBrowser : public QLuaMainWindow
{
  Q_OBJECT
  Q_PROPERTY(QUrl url READ url WRITE setUrl)
  Q_PROPERTY(QUrl homeUrl READ homeUrl WRITE setHomeUrl)
  Q_PROPERTY(QString baseTitle READ baseTitle WRITE setBaseTitle)
  Q_PROPERTY(QString pageTitle READ pageTitle)
  Q_PROPERTY(QString html READ toHtml WRITE setHtml)
public:
  QLuaBrowser(QWidget *parent=0);
  QUrl url() const;
  QUrl homeUrl() const;
  QString baseTitle() const;
  QString pageTitle() const;
  QString toHtml() const;
  Q_INVOKABLE QWebView *view();
  Q_INVOKABLE QWebPage *page();
public slots:
  void setUrl(QUrl url);
  void setHomeUrl(QUrl url);
  void setBaseTitle(QString s);
  void setHtml(QString html);
  virtual bool openFile(QString fileName, bool inOther=false);
  virtual bool newDocument();
  virtual void doOpenLocation();
  virtual void doOpenBrowser();
  virtual void doSaveAs();
  virtual void doPrint();
  virtual void doCopy();
  virtual void doEdit();
  virtual void doFind();
  virtual void doHome();
  virtual void doForward();
  virtual void doBackward();
  virtual void doStop();
  virtual void doReload();
  virtual void doZoomIn();
  virtual void doZoomOut();
  virtual void doZoomReset();
  virtual void updateWindowTitle();
  virtual void updateWindowIcon();
  virtual void updateActions();
public:
  virtual QAction *createAction(QByteArray);
  virtual QToolBar *createToolBar();
  virtual QMenuBar  *createMenuBar();
  virtual QStatusBar *createStatusBar();
  virtual void loadSettings();
  virtual void saveSettings();
  class Private;
  class WebView;
  class FindDialog;
private:
  Private *d;
};

#endif



/* -------------------------------------------------------------
   Local Variables:
   c++-font-lock-extra-types: ("\\sw+_t" "\\(lua_\\)?[A-Z]\\sw*[a-z]\\sw*")
   End:
   ------------------------------------------------------------- */
