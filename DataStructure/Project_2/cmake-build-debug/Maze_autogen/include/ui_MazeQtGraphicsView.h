/********************************************************************************
** Form generated from reading UI file 'MazeQtGraphicsView.ui'
**
** Created by: Qt User Interface Compiler version 6.6.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAZEQTGRAPHICSVIEW_H
#define UI_MAZEQTGRAPHICSVIEW_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QGraphicsView>

QT_BEGIN_NAMESPACE

class Ui_MazeQGraphicsView
{
public:

    void setupUi(QGraphicsView *MazeQGraphicsView)
    {
        if (MazeQGraphicsView->objectName().isEmpty())
            MazeQGraphicsView->setObjectName("MazeQGraphicsView");
        MazeQGraphicsView->resize(400, 300);

        retranslateUi(MazeQGraphicsView);

        QMetaObject::connectSlotsByName(MazeQGraphicsView);
    } // setupUi

    void retranslateUi(QGraphicsView *MazeQGraphicsView)
    {
        MazeQGraphicsView->setWindowTitle(QCoreApplication::translate("MazeQGraphicsView", "MazeQGraphicsView", nullptr));
    } // retranslateUi

};

namespace Ui {
    class MazeQGraphicsView: public Ui_MazeQGraphicsView {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAZEQTGRAPHICSVIEW_H
