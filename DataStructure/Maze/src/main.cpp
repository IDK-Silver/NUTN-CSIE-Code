#include <QtWidgets/QApplication>
#include <src/widget/MainWindow/MainWindow.h>
int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    MainWindow window;
    window.show();
    return QApplication::exec();
}