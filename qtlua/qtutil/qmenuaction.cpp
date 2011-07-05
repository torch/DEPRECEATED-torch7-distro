

#include "qmenuaction.h"

#include <QMap>


struct QMenuAction::Private
{
  QList<QAction*> actions;
  QMap<QAction*,QAction*> translate;
  QList<QAction*> aliases;
  QList<QAction*> owned;
};


QMenuAction::QMenuAction(QObject *parent)
  : QAction(parent), d(new Private)
{
  connect(this, SIGNAL(changed()), this, SLOT(p_changed()));
}


QMenuAction::QMenuAction(const QString &text, QObject *parent)
  : QAction(text, parent), d(new Private)
{
  connect(this, SIGNAL(changed()), this, SLOT(p_changed()));
}


QMenuAction::QMenuAction(const QIcon &icon, const QString &text, QObject *parent)
  : QAction(icon, text, parent), d(new Private)
{
  connect(this, SIGNAL(changed()), this, SLOT(p_changed()));
}


QMenuAction::~QMenuAction()
{
  removeEventFilter(this);
  clear();
  delete d;
  d = 0;
}


void 
QMenuAction::addAction(QAction *action)
{
  if (action)
    d->actions += action;
}


void 
QMenuAction::addActions(QList<QAction*> actions)
{
  foreach(QAction *action, actions)
    if (action)
      d->actions += action;
}


QList<QAction*> 
QMenuAction::actions() const
{
  return d->actions;
}


QAction *
QMenuAction::addAction(const QString &text)
{
  QAction *a = new QAction(text, this);
  d->owned += a;
  addAction(a);
  return a;
}


QAction *
QMenuAction::addAction(const QIcon &icon, const QString &text)
{
  QAction *a = new QAction(icon, text, this);
  d->owned += a;
  addAction(a);
  return a;
}


QAction *
QMenuAction::addAction(const QString &text, 
                       const QObject *receiver, const char* member, 
                       const QKeySequence &shortcut)
{
  QAction *a = new QAction(text, this);
  connect(a, SIGNAL(triggered(bool)), receiver, member);
  if (&shortcut) a->setShortcut(shortcut);
  d->owned += a;
  addAction(a);
  return a;
}


QAction *
QMenuAction::addAction(const QIcon &icon, const QString &text, 
                       const QObject *receiver, const char* member, 
                       const QKeySequence &shortcut)
{
  QAction *a = new QAction(icon, text, this);
  connect(a, SIGNAL(triggered(bool)), receiver, member);
  if (&shortcut) a->setShortcut(shortcut);
  d->owned += a;
  addAction(a);
  return a;
}


QMenuAction *
QMenuAction::addMenu(QMenuAction *menu)
{
  addAction(menu);
  return menu;
}


QMenuAction *
QMenuAction::addMenu(const QString &title)
{
  QMenuAction *a = new QMenuAction(title, this);
  d->owned += a;
  addAction(a);
  return a;
}


QMenuAction *
QMenuAction::addMenu(const QIcon &icon, const QString &title)
{
  QMenuAction *a = new QMenuAction(icon, title, this);
  d->owned += a;
  addAction(a);
  return a;
}


QAction *
QMenuAction::addSeparator()
{
  QAction *a = new QAction(this);
  a->setSeparator(true);
  d->owned += a;
  addAction(a);
  return a;
}


void 
QMenuAction::insertAction(QAction *before, QAction *action)
{
  int i = d->actions.indexOf(before);
  if (i < 0)
    i = d->actions.size();
  if (action)
    d->actions.insert(i, action);
}


void 
QMenuAction::insertActions(QAction *before, QList<QAction*> actions)
{
  int i = d->actions.indexOf(before);
  if (i < 0)
    i = d->actions.size();
  foreach(QAction *action, actions)
    if (action)
      d->actions.insert(i++, action);
}


QMenuAction *
QMenuAction::insertMenu(QAction *before, QMenuAction *menu)
{
  insertAction(before, menu);
  return menu;
}


QAction *
QMenuAction::insertSeparator(QAction *before)
{
  QAction *a = new QAction(this);
  a->setSeparator(true);
  d->owned += a;
  insertAction(before, a);
  return a;
}


void 
QMenuAction::removeAction(QAction *action)
{
  d->actions.removeAll(action);
  if (d->owned.contains(action))
    {
      d->owned.removeAll(action);
      delete action;
    }
}


bool 
QMenuAction::isEmpty() const
{
  return d->actions.isEmpty();
}


void 
QMenuAction::clear()
{
  d->actions.clear();
  foreach(QAction *a, d->owned)
    delete a;
  d->owned.clear();
}


QAction*
QMenuAction::menuAction()
{
  return this;
}


void 
QMenuAction::setTitle(const QString &title)
{
  setText(title);
}


QString 
QMenuAction::title() const
{
  return text();
}


template<class T> 
class AutoDelete
{ 
  T* p;
public:
  AutoDelete(T* p) : p(p) {}
  ~AutoDelete() { delete p; }
  operator T*() const { return p; }
  T& operator*() const { return *p; }
  T* operator->() const { return p; }
private:
  AutoDelete(const AutoDelete&);
  AutoDelete& operator=(const AutoDelete&);
};
  

QAction *
QMenuAction::exec(const QPoint &point, QAction *action)
{
  AutoDelete<QMenu> m(buildMenu());
  return m->exec();
}


void 
QMenuAction::copyMenu(QWidget *menu)
{
  foreach(QAction *action, d->actions)
    {
      QMenuAction *maction = qobject_cast<QMenuAction*>(action);
      if (maction)
        {
          QAction *raction = maction->buildMenu(menu)->menuAction();
          connect(raction, SIGNAL(destroyed(QObject*)), this, SLOT(p_destroyed(QObject*)));
          connect(raction, SIGNAL(hovered()), this, SIGNAL(hovered()));
          connect(raction, SIGNAL(toggled(bool)), this, SIGNAL(toggled(bool)));
          connect(raction, SIGNAL(triggered(bool)), this, SIGNAL(triggered(bool)));
          d->translate[raction] = maction;
          menu->addAction(raction);
        }
      else
        menu->addAction(action);
    }
}                                              


QMenuBar *
QMenuAction::buildMenuBar(QWidget *parent)
{
  QMenuBar *menu = new QMenuBar(parent);
  connect(menu, SIGNAL(hovered(QAction*)), this, SLOT(p_hovered(QAction*)));
  connect(menu, SIGNAL(triggered(QAction*)), this, SLOT(p_triggered(QAction*)));
  copyMenu(menu);
  return menu;
}


QMenu *
QMenuAction::buildMenu(QWidget *parent)
{
  QMenu *menu = new QMenu(parent);
  connect(menu, SIGNAL(aboutToHide()), this, SIGNAL(aboutToHide()));
  connect(menu, SIGNAL(aboutToShow()), this, SIGNAL(aboutToShow()));
  connect(menu, SIGNAL(hovered(QAction*)), this, SLOT(p_hovered(QAction*)));
  connect(menu, SIGNAL(triggered(QAction*)), this, SLOT(p_triggered(QAction*)));
  QAction *menuAction = menu->menuAction();
  connect(menuAction, SIGNAL(destroyed(QObject*)), this, SLOT(p_destroyed(QObject*)));
  d->aliases += menuAction;
  syncAction(this, menuAction);
  menu->setTitle(text());
  copyMenu(menu);
  return menu;
}


void 
QMenuAction::p_destroyed(QObject *o)
{
  QAction *a = qobject_cast<QAction*>(o);
  if (a) 
    {
      d->aliases.removeAll(a);
      d->translate.remove(a);
    }
}


void 
QMenuAction::p_hovered(QAction *a)
{
  QMap<QAction*,QAction*>::const_iterator it = d->translate.find(a);
  if (it != d->translate.end()) a = *it;
  emit hovered(a);
}


void 
QMenuAction::p_triggered(QAction *a)
{
  QMap<QAction*,QAction*>::const_iterator it = d->translate.find(a);
  if (it != d->translate.end()) a = *it;
  emit triggered(a);
}

  
void 
QMenuAction::syncAction(QAction *from, QAction *to)
{
  to->setText(from->text());
  to->setIcon(from->icon());
  to->setData(from->data());
  to->setIconText(from->iconText());
  to->setEnabled(from->isEnabled());
  to->setVisible(from->isVisible());
  to->setShortcuts(from->shortcuts());
  to->setShortcutContext(from->shortcutContext());
  to->setStatusTip(from->statusTip());
  to->setToolTip(from->toolTip());
}


void
QMenuAction::p_changed()
{
  foreach(QAction *alias, d->aliases)
    syncAction(this, alias);
}



/* -------------------------------------------------------------
   Local Variables:
   c++-font-lock-extra-types: ("\\sw+_t" "\\(lua_\\)?[A-Z]\\sw*[a-z]\\sw*")
   End:
   ------------------------------------------------------------- */
