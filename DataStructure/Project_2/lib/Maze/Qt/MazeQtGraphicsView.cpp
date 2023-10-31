//
// Created by idk on 2023/10/30.
//

#include "MazeQtGraphicsView.h"
#include <QMouseEvent>
#include <QGraphicsView>
#include <QGraphicsScene>
#include <QGraphicsPixmapItem>
#include <QPixmap>
#include <iostream>

MazeQtGraphicsView::MazeQtGraphicsView(QWidget *parent) : QGraphicsView(parent) {

    // create scene to add item to graphics view
    this->scene = new QGraphicsScene(this);

    // set scene
    this->setScene(this->scene);

    // create maze data
    this->maze = std::make_shared<QVector<QVector<MazeQtGraphicsPixmapItem*>>>();


}



void MazeQtGraphicsView::mousePressEvent(QMouseEvent *event) {

    /* if not painting mode not change maze obj */
    if (!this->isPaintingMode()) {
        QGraphicsView::mousePressEvent(event);
        return;
    }

    MazeObject changeObj = MazeObject::Blank;

    if (this->_nextIsSetStartPoint) {
        changeObj = MazeObject::Start;
    }

    if (this->_nextIsSetEndPoint) {
        changeObj = MazeObject::End;
    }


    // Get the position of the click event.
    QPointF clickPos = mapToScene(event->pos());

    // Use scene()->items() to find items at the click position.
    QList<QGraphicsItem *> items = this->scene->items(clickPos);

    for (auto item : items)
    {
        if (item->type() == MazeQtGraphicsPixmapItem::Type)
        {
            // Change the content of the QGraphicsPixmapItem.
            auto *pixmapItem = dynamic_cast<MazeQtGraphicsPixmapItem *>(item);

            size_t  row_index = pixmapItem->getIndex().first,
                    column_index = pixmapItem->getIndex().second;

            // set maze obj type
            pixmapItem->setType(changeObj);
            this->originMaze->at(row_index, column_index) = changeObj;

            // emit qt Signal of completed start point
            if (_nextIsSetStartPoint)
                    emit completedStartPoint(row_index, column_index);

            // emit qt Signal of completed end point
            if (_nextIsSetEndPoint)
                    emit completedEndPoint(row_index, column_index);
        }
    }

    /* restore flag */
    this->_nextIsSetEndPoint = false;
    this->_nextIsSetStartPoint = false;

    /* print maze */
//    for (size_t row = 0; row < originMaze->getSize().first; row++) {
//        qDebug() << originMaze->at(row);
//    }

    // Call the base class implementation to ensure normal event handling.
    QGraphicsView::mousePressEvent(event);
}

void MazeQtGraphicsView::createMaze(std::shared_ptr<Maze> obj) {

    // storage origin maze
    this->originMaze = obj;

    // clear old map
    this->maze->clear();

    // delete old maze obj
    for (auto item : this->scene->items()) {
        delete(item);
    }

    // create maze
    for (int row = 0; row < obj->getSize().first; row++) {

        QVector<MazeQtGraphicsPixmapItem *> row_vec;

        for (int column = 0; column < obj->getSize().second; column++) {

            // create each maze obj
            auto element = new MazeQtGraphicsPixmapItem();

            // set index
            element->setIndex(row, column);

            // set maze obj type
            element->setType(obj->at(row).at(column));

            // set initial pos
            element->setPos(column * element->getSize().width(), row * element->getSize().height());

            // add maze obj to scene
            this->scene->addItem(element);

            // add row vector to maze data
            row_vec.push_back(element);
        }
        this->maze->push_back(row_vec);
    }


    // fit graphics window
    fitInView(this->scene->sceneRect(), Qt::KeepAspectRatio);
}

MazeQtGraphicsView::~MazeQtGraphicsView() {
    delete(this->scene);
}

MazeObject MazeQtGraphicsView::get(size_t row, size_t column) {
    return this->maze->at(row).at(column)->getType();
}

void MazeQtGraphicsView::setPaintingMode(bool flag) {
    this->_isPaintingMode = flag;
}

bool MazeQtGraphicsView::isPaintingMode() const {
    return this->_isPaintingMode;
}

void MazeQtGraphicsView::setNextIsStartPoint() {
    this->_nextIsSetStartPoint = true;
}

void MazeQtGraphicsView::setNextIsEndPoint() {
    this->_nextIsSetEndPoint = true;

}

