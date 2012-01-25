// -*- C++ -*-

#ifndef QMENUACTION_H
#define QMENUACTION_H

#include "qtutilconf.h"

#include <QAction>
#include <QIcon>
#include <QKeySequence>
#include <QList>
#include <QMenu>
#include <QMenuBar>
#include <QObject>
#include <QPoint>
#include <QString>


QTUTILAPI class QMenuAction : public QAction 
{
  Q_OBJECT
  
public:
  QMenuAction(QObject *parent = 0);
  QMenuAction(const QString &text, QObject *parent);
  QMenuAction(const QIcon &icon, const QString &text, QObject *parent);
  ~QMenuAction();
  
  QMenuBar *buildMenuBar(QWidget *parent = 0);
  QMenu *buildMenu(QWidget *parent = 0);
  
  void addAction(QAction *action);
  void addActions(QList<QAction*> actions);
  QList<QAction*> actions() const;

  QAction *addAction(const QString &text);
  QAction *addAction(const QIcon &icon, const QString &text);
  QAction *addAction(const QString &text, 
                     const QObject *receiver, const char* member, 
                     const QKeySequence &shortcut = 0);
  QAction *addAction(const QIcon &icon, const QString &text, 
                     const QObject *receiver, const char* member, 
                     const QKeySequence &shortcut = 0);

  QMenuAction *addMenu(QMenuAction *menu);
  QMenuAction *addMenu(const QString &title);
  QMenuAction *addMenu(const QIcon &icon, const QString &title);
  QAction *addSeparator();
  
  void insertAction(QAction *before, QAction *action);
  void insertActions(QAction *before, QList<QAction*> actions);
  QMenuAction *insertMenu(QAction *before, QMenuAction *menu);
  QAction *insertSeparator(QAction *before);

  void removeAction(QAction *action);

  bool isEmpty() const;
  void clear();
  QAction *menuAction();
  void setTitle(const QString &title);
  QString title () const;
  QAction *exec(const QPoint &point, QAction *action = 0);

signals:
  void aboutToHide ();
  void aboutToShow ();
  void hovered(QAction *action);
  void triggered(QAction *action);

protected:
  void syncAction(QAction *from, QAction *to);
  void copyMenu(QWidget *menu);                                                  

protected slots:
  void p_destroyed(QObject*);
  void p_hovered(QAction*);
  void p_triggered(QAction*);
  void p_changed();

public:
  struct Private;  
private:
  Private *d;
};



#endif
/* -------------------------------------------------------------
   Local Variables:
   c++-font-lock-extra-types: ("\\sw+_t" "\\(lua_\\)?[A-Z]\\sw*[a-z]\\sw*")
   End:
   ------------------------------------------------------------- */

