/* -*- C++ -*- */


#include "qtluaprinter.h"

#include <QDialog>
#include <QPrintDialog>



struct Option 
{
  const char *name; 
  int value;
};

#define F(t) {#t, (int) QPrinter::t}

static Option pageSizes[] = {
  F(A4), F(B5), F(Letter), F(Legal), F(Executive),
  F(A0), F(A1), F(A2), F(A3), F(A5), F(A6), F(A7), F(A8), F(A9), F(B0), F(B1),
  F(B10), F(B2), F(B3), F(B4), F(B6), F(B7), F(B8), F(B9), F(C5E), F(Comm10E),
  F(DLE), F(Folio), F(Ledger), F(Tabloid), F(Custom),
  {0} };

static Option outputFormats[] = {
  F(NativeFormat), F(PdfFormat), F(PostScriptFormat), 
  {0} };

static Option printerStates[] = {
  F(Idle), F(Active), F(Aborted), F(Error),
  {0} };


static const char *
value_to_name(int value, Option *opts)
{
  for(; opts->name; opts++)
    if  (opts->value == value)
      return opts->name;
  return "unknown";
}


static int
name_to_value(const char *name, Option *opts)
{
  for(; opts->name; opts++)
    if  (! strcmp(name, opts->name))
      return opts->value;
  return -1;
}


QString 
QtLuaPrinter::pageSize() const
{
#if QT_VERSION >= 0x40400
  int s = (int)QPrinter::paperSize();
#else
  int s = (int)QPrinter::pageSize();
#endif
  return QString::fromAscii(value_to_name(s, pageSizes));
}


void 
QtLuaPrinter::setPageSize(QString r)
{
  int s = name_to_value(r.toLocal8Bit().constData(), pageSizes);
  if (s >= 0)
    {
      custom = false;
      if (s != QPrinter::Custom)
        QPrinter::setPageSize(QPrinter::PageSize(s));
#if QT_VERSION >= 0x40400
      else 
        custom = true;
      if (custom && papSize.isValid())
        QPrinter::setPaperSize(papSize, Point); 
#endif
    }
}


QSizeF 
QtLuaPrinter::paperSize() const 
{
  return papSize;
}


void 
QtLuaPrinter::setPaperSize(QSizeF s) 
{ 
  papSize = s; 
#if QT_VERSION >= 0x40400
  if (custom && papSize.isValid())
    QPrinter::setPaperSize(papSize, Point); 
#endif
}


QString 
QtLuaPrinter::outputFormat() const
{
  int s = (int) QPrinter::outputFormat();
  return QString::fromAscii(value_to_name(s, outputFormats));
}


void 
QtLuaPrinter::setOutputFormat(QString r)
{
  int s = name_to_value(r.toLocal8Bit().constData(), outputFormats);
  if (s >= 0)
    QPrinter::setOutputFormat(QPrinter::OutputFormat(s));
}


QString 
QtLuaPrinter::printerState() const
{
  int s = (int) QPrinter::printerState();
  return QString::fromAscii(value_to_name(s, printerStates));
}


bool
QtLuaPrinter::setup(QWidget *parent)
{
  QPointer<QPrintDialog> dialog = new QPrintDialog(this, parent);
  dialog->setFromTo(fromPage(), toPage());
  // options
  dialog->addEnabledOption(QPrintDialog::PrintToFile);
  dialog->addEnabledOption(QPrintDialog::PrintPageRange);
  dialog->addEnabledOption(QPrintDialog::PrintCollateCopies);
  // exec
  int result = dialog->exec();
  delete dialog;
  return (result == QDialog::Accepted);
}


QtLuaPrinter::~QtLuaPrinter()
{
  emit closing(this);
}






/* -------------------------------------------------------------
   Local Variables:
   c++-font-lock-extra-types: ("\\sw+_t" "\\(lua_\\)?[A-Z]\\sw*[a-z]\\sw*" "qreal")
   End:
   ------------------------------------------------------------- */
