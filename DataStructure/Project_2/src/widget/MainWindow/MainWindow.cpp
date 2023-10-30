//
// Created by idk on 2023/10/25.
//

#include "MainWindow.h"
#include "./ui_MainWindow.h"


MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent), ui(new Ui::MainWindow)
{

    // init widget object
    this->ui->setupUi(this);

//    // init maze graphics view
//    this->maze_view = std::make_shared<MazeQtGraphics>(this->ui->mazeView_graphicsView);
//
//    auto scene = new QGraphicsScene();
//    auto view =  this->ui->mazeView_graphicsView;
//    view->setScene(scene);
//
//    // 添加迷宮元素到場景中
//    QGraphicsRectItem *wall = new QGraphicsRectItem(50, 50, 100, 100);
//    wall->setBrush(QBrush(Qt::black));
//    scene->addItem(wall);
//
//    QGraphicsEllipseItem *player = new QGraphicsEllipseItem(75, 75, 10, 10);
//    player->setBrush(QBrush(Qt::red));
//    scene->addItem(player);
//
//    // 創建一條直線
//    QGraphicsLineItem *line = new QGraphicsLineItem(50, 50, 200, 200);
//
//    // 設置線條顏色和寬度
//    QPen pen;
//    pen.setColor(Qt::blue);
//    pen.setWidth(2);
//    line->setPen(pen);
//    scene->addItem(line);
//    line->setLine(100, 100, 300, 1000);
////    view->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOn);
////    view->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOn);
//
//
//    // 設置視圖視口
//    view->setRenderHint(QPainter::Antialiasing);
//    view->setViewportUpdateMode(QGraphicsView::BoundingRectViewportUpdate);
//    view->setRenderHint(QPainter::Antialiasing);
//    view->setBackgroundBrush(QBrush(Qt::white));
//    // view->setFixedSize(1000, 1000);
//    view->show();
}

MainWindow::~MainWindow() = default;
