//
// Created by idk on 2023/10/25.
//

#include <iostream>
#include "MainWindow.h"
#include "./ui_MainWindow.h"


MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent), ui(new Ui::MainWindow)
{

    // init widget object
    this->ui->setupUi(this);

    // initial Slot
    this->connectSetUp();


    // set choose maze start end spin box can't change value by user
    this->ui->mazeStartPoint_X_spinBox->setReadOnly(true);
    this->ui->mazeStartPoint_Y_spinBox->setReadOnly(true);
    this->ui->mazeEndPoint_X_spinBox->setReadOnly(true);
    this->ui->mazeEndPoint_Y_spinBox->setReadOnly(true);
}

void MainWindow::createMaze() {


    // create maze
    this->maze = std::make_shared<Maze>(
            this->ui->mazeM_spinBox->value(),
            this->ui->mazeN_spinBox->value());

    // create and initial maze obj, push into maze
    for (int row = 0; row < maze->getSize().first; row++) {

        for (int column = 0; column < maze->getSize().second; column++) {
            maze->at(row).at(column) = MazeObject::Blank;
        }
    }

    // make graphics view to show maze
    this->ui->mazeView_graphicsView->createMaze(maze);

    // make graphics view can change wall obj by mouse press *.
    this->ui->mazeView_graphicsView->setPaintingMode(true);

    // set choose maze start end spin box maximum
    this->ui->mazeStartPoint_X_spinBox->setMaximum(this->ui->mazeM_spinBox->value() - 1);
    this->ui->mazeStartPoint_Y_spinBox->setMaximum(this->ui->mazeN_spinBox->value() - 1);
    this->ui->mazeEndPoint_X_spinBox->setMaximum(this->ui->mazeM_spinBox->value() - 1);
    this->ui->mazeEndPoint_Y_spinBox->setMaximum(this->ui->mazeN_spinBox->value() - 1);
}

/* the function to initial Qt Slot connect */
void MainWindow::connectSetUp() {

    // connect Maze Size Confirm Push Button to createMaze
    connect(this->ui->mazeSizeConfirm_pushButton, SIGNAL(clicked()), SLOT(createMaze()));

    // connect choose maze start point to setting maze graphics view to mark maze obj
    connect(this->ui->chooseStartPoint_pushButton, &QPushButton::clicked, [=]() {
        this->ui->mazeView_graphicsView->setNextIsStartPoint();
    });

    // connect choose maze end point to setting maze graphics view to mark maze obj
    connect(this->ui->chooseEndPoint_pushButton, &QPushButton::clicked, [=]() {
        this->ui->mazeView_graphicsView->setNextIsEndPoint();
    });


    connect(this->ui->mazeView_graphicsView, &MazeQtGraphicsView::completedStartPoint, [=](size_t row, size_t column) {
        this->ui->mazeStartPoint_X_spinBox->setValue(row);
        this->ui->mazeStartPoint_Y_spinBox->setValue(column);
    });

    connect(this->ui->mazeView_graphicsView, &MazeQtGraphicsView::completedEndPoint, [=](size_t row, size_t column) {
        this->ui->mazeEndPoint_X_spinBox->setValue(row);
        this->ui->mazeEndPoint_Y_spinBox->setValue(column);
    });


//    connect(this->ui->mazeView_graphicsView, SIGNAL(completedStartPoint), SLOT(chooseStartPoint));
//    connect(this->ui->mazeView_graphicsView)
}


MainWindow::~MainWindow() = default;
