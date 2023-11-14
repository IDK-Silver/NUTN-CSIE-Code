//
// Created by idk on 2023/10/25.
//

#include <iostream>
#include "MainWindow.h"
#include "./ui_MainWindow.h"
#include <lib/Maze/MazeAlgorithm.h>
#include <QMessageBox>

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent), ui(new Ui::MainWindow)
{

    // init widget object
    this->ui->setupUi(this);

    // initial Slot
    this->connectSetUp();


    // set painting group is not enable
    this->ui->PaintingGroup->setEnabled(false);

    // set choose maze start end spin box can't change value by user
    this->ui->mazeStartPoint_X_spinBox->setReadOnly(true);
    this->ui->mazeStartPoint_Y_spinBox->setReadOnly(true);
    this->ui->mazeEndPoint_X_spinBox->setReadOnly(true);
    this->ui->mazeEndPoint_Y_spinBox->setReadOnly(true);


}

void MainWindow::createMaze() {

    // set painting group enable
    this->ui->PaintingGroup->setEnabled(true);



    // create maze
    this->maze = std::make_shared<Maze>(
            this->ui->mazeM_spinBox->value(),
            this->ui->mazeN_spinBox->value());

    // create and initial maze obj, push into maze
    for (int row = 0; row < maze->getSize().first; row++) {

        for (int column = 0; column < maze->getSize().second; column++) {
            maze->at(row, column) = MazeObject::Wall;
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

    // Qt widget init
    this->ui->mazeStartPoint_X_spinBox->setValue(0);
    this->ui->mazeStartPoint_Y_spinBox->setValue(0);
    this->ui->mazeEndPoint_X_spinBox->setValue(0);
    this->ui->mazeEndPoint_Y_spinBox->setValue(0);
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

    connect(this->ui->mazePaintConfirm_pushButton, &QPushButton::clicked, [=]() {


        // if not create maze
        if (this->maze == nullptr) {
            QMessageBox::warning(nullptr, "提示", "請先創建地圖");
            return;
        }

        // clear the last road hit
        this->ui->mazeView_graphicsView->clearRoadHit();

        // get new road hit
        auto result = MazeAlgorithm::getRoadHit(this->maze);


        /* if maze solution is not exits */
        if (result.size() <= 2 &&
            !(result.size() == 2 &&
             result.at(0).x == this->ui->mazeStartPoint_X_spinBox->value() &&
             result.at(0).y == this->ui->mazeStartPoint_Y_spinBox->value() &&
             result.at(1).x == this->ui->mazeStartPoint_X_spinBox->value() &&
             result.at(1).y == this->ui->mazeStartPoint_Y_spinBox->value())
                )
        {
            QMessageBox::warning(nullptr, "提示", "迷宮無解");
            return;
        }

        // show the road hit
        for (auto road_hit_point : result) {

            this->ui->mazeView_graphicsView->setMazeObject(road_hit_point.x,
                                                           road_hit_point.y,
                                                           MazeObject::RoadHint);
        }

        // reset start road hit
        this->ui->mazeView_graphicsView->setMazeObject(this->ui->mazeStartPoint_X_spinBox->value(),
                                                       this->ui->mazeStartPoint_Y_spinBox->value(),
                                                       MazeObject::Start);

        //  show how many step that star point to end point
        QMessageBox::information(nullptr, "提示", QString("迷宮所需步數 : %1").arg(result.size() -2));

    });

//    connect(this->ui->mazeView_graphicsView, SIGNAL(completedStartPoint), SLOT(chooseStartPoint));
//    connect(this->ui->mazeView_graphicsView)
}


MainWindow::~MainWindow() = default;
