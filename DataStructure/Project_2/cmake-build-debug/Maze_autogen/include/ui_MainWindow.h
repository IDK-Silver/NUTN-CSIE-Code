/********************************************************************************
** Form generated from reading UI file 'MainWindow.ui'
**
** Created by: Qt User Interface Compiler version 6.6.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINWINDOW_H
#define UI_MAINWINDOW_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenu>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>
#include <lib/Maze/Qt/MazeQtGraphicsView.h>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QWidget *centralwidget;
    QHBoxLayout *horizontalLayout;
    QVBoxLayout *verticalLayout;
    QGroupBox *groupBox;
    QVBoxLayout *verticalLayout_3;
    QVBoxLayout *verticalLayout_2;
    QGridLayout *gridLayout_3;
    QLabel *label;
    QSpinBox *mazeM_spinBox;
    QLabel *label_2;
    QSpinBox *mazeN_spinBox;
    QPushButton *mazeSizeConfirm_pushButton;
    QGroupBox *groupBox_2;
    QGridLayout *gridLayout;
    QPushButton *pushButton;
    MazeQtGraphicsView *mazeView_graphicsView;
    QMenuBar *menubar;
    QMenu *menu;
    QMenu *menu_2;
    QStatusBar *statusbar;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName("MainWindow");
        MainWindow->resize(924, 585);
        centralwidget = new QWidget(MainWindow);
        centralwidget->setObjectName("centralwidget");
        horizontalLayout = new QHBoxLayout(centralwidget);
        horizontalLayout->setObjectName("horizontalLayout");
        verticalLayout = new QVBoxLayout();
        verticalLayout->setObjectName("verticalLayout");
        groupBox = new QGroupBox(centralwidget);
        groupBox->setObjectName("groupBox");
        verticalLayout_3 = new QVBoxLayout(groupBox);
        verticalLayout_3->setObjectName("verticalLayout_3");
        verticalLayout_2 = new QVBoxLayout();
        verticalLayout_2->setObjectName("verticalLayout_2");
        gridLayout_3 = new QGridLayout();
        gridLayout_3->setObjectName("gridLayout_3");
        label = new QLabel(groupBox);
        label->setObjectName("label");

        gridLayout_3->addWidget(label, 0, 0, 1, 1);

        mazeM_spinBox = new QSpinBox(groupBox);
        mazeM_spinBox->setObjectName("mazeM_spinBox");
        mazeM_spinBox->setMinimum(1);
        mazeM_spinBox->setMaximum(1000);
        mazeM_spinBox->setValue(100);

        gridLayout_3->addWidget(mazeM_spinBox, 0, 1, 1, 1);

        label_2 = new QLabel(groupBox);
        label_2->setObjectName("label_2");

        gridLayout_3->addWidget(label_2, 1, 0, 1, 1);

        mazeN_spinBox = new QSpinBox(groupBox);
        mazeN_spinBox->setObjectName("mazeN_spinBox");
        mazeN_spinBox->setMinimum(1);
        mazeN_spinBox->setMaximum(100);
        mazeN_spinBox->setValue(100);

        gridLayout_3->addWidget(mazeN_spinBox, 1, 1, 1, 1);


        verticalLayout_2->addLayout(gridLayout_3);


        verticalLayout_3->addLayout(verticalLayout_2);

        mazeSizeConfirm_pushButton = new QPushButton(groupBox);
        mazeSizeConfirm_pushButton->setObjectName("mazeSizeConfirm_pushButton");

        verticalLayout_3->addWidget(mazeSizeConfirm_pushButton);

        groupBox_2 = new QGroupBox(groupBox);
        groupBox_2->setObjectName("groupBox_2");
        gridLayout = new QGridLayout(groupBox_2);
        gridLayout->setObjectName("gridLayout");
        pushButton = new QPushButton(groupBox_2);
        pushButton->setObjectName("pushButton");

        gridLayout->addWidget(pushButton, 0, 0, 1, 1);


        verticalLayout_3->addWidget(groupBox_2);


        verticalLayout->addWidget(groupBox);


        horizontalLayout->addLayout(verticalLayout);

        mazeView_graphicsView = new MazeQtGraphicsView(centralwidget);
        mazeView_graphicsView->setObjectName("mazeView_graphicsView");

        horizontalLayout->addWidget(mazeView_graphicsView);

        MainWindow->setCentralWidget(centralwidget);
        menubar = new QMenuBar(MainWindow);
        menubar->setObjectName("menubar");
        menubar->setGeometry(QRect(0, 0, 924, 27));
        menu = new QMenu(menubar);
        menu->setObjectName("menu");
        menu_2 = new QMenu(menubar);
        menu_2->setObjectName("menu_2");
        MainWindow->setMenuBar(menubar);
        statusbar = new QStatusBar(MainWindow);
        statusbar->setObjectName("statusbar");
        MainWindow->setStatusBar(statusbar);

        menubar->addAction(menu_2->menuAction());
        menubar->addAction(menu->menuAction());

        retranslateUi(MainWindow);

        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QCoreApplication::translate("MainWindow", "\350\277\267\345\256\256\346\216\242\347\264\242", nullptr));
        groupBox->setTitle(QCoreApplication::translate("MainWindow", "\350\277\267\345\256\256\350\250\255\345\256\232\345\215\200\345\237\237", nullptr));
        label->setText(QCoreApplication::translate("MainWindow", "\351\225\267\345\272\246 (M)", nullptr));
        label_2->setText(QCoreApplication::translate("MainWindow", "\345\257\254\345\272\246(N)", nullptr));
        mazeSizeConfirm_pushButton->setText(QCoreApplication::translate("MainWindow", "\347\242\272\350\252\215\350\250\255\345\256\232", nullptr));
        groupBox_2->setTitle(QCoreApplication::translate("MainWindow", "\345\212\237\350\203\275\346\214\211\351\210\225", nullptr));
        pushButton->setText(QCoreApplication::translate("MainWindow", "PushButton", nullptr));
        menu->setTitle(QCoreApplication::translate("MainWindow", "\350\252\252\346\230\216", nullptr));
        menu_2->setTitle(QCoreApplication::translate("MainWindow", "\346\252\224\346\241\210", nullptr));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
