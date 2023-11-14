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

            // if select start point or end point to change
            // and it not the set start point or end point event
            // then cancel
            if (!_nextIsSetStartPoint && !_nextIsSetEndPoint) {
                if (pixmapItem->getIndex() == start_point || pixmapItem->getIndex() == end_point)  {
                    return;
                }
            }


            // if clicked obj is blank change obj to wall
            if (this->_nextIsSetStartPoint) {

                /* change last start point to wall */
                if (start_point != unset_point)
                {
                    // change origin obj
                    this->originMaze->at(this->start_point.first, this->start_point.second) = Wall;
                    // change graphics view  obj
                    this->setMazeObject(this->start_point.first, this->start_point.second, Wall);
                }


                // set obj type
                changeObj = MazeObject::Start;
            }
            else if (this->_nextIsSetEndPoint) {

                /* change last start point to wall */
                if (end_point != unset_point)
                {
                    // change origin obj
                    this->originMaze->at(this->end_point.first, this->end_point.second) = Wall;
                    // change graphics view  obj
                    this->setMazeObject(this->end_point.first, this->end_point.second, Wall);
                }


                // set obj type
                changeObj = MazeObject::End;
            }

            // change blank to wall
            else if (pixmapItem->getType() == Blank) {
                changeObj = Wall;
            }

            // set maze obj type
            pixmapItem->setType(changeObj);
            this->originMaze->at(row_index, column_index) = changeObj;

            // change success event
            if (_nextIsSetStartPoint) {

                // set last start point
                start_point = std::pair<size_t , size_t >(row_index, column_index);

                // emit qt Signal of completed start point
                emit completedStartPoint(row_index, column_index);
            }

            // change success event
            if (_nextIsSetEndPoint) {

                // set last start point
                end_point = std::pair<size_t , size_t >(row_index, column_index);

                // emit qt Signal of completed end point
                emit completedEndPoint(row_index, column_index);
            }

        }
    }

    /* restore flag */
    this->_nextIsSetEndPoint = false;
    this->_nextIsSetStartPoint = false;

    QGraphicsView::mousePressEvent(event);
}

void MazeQtGraphicsView::createMaze(std::shared_ptr<Maze> obj) {

    // reset start point and end point
    this->start_point = unset_point;
    this->end_point = unset_point;

    // delete old scene
    if (this->scene != nullptr) {
        this->scene->clear();
        delete(this->scene);
    }

    // create new scene
    this->scene = new QGraphicsScene(this);
    this->setScene(this->scene);

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

void MazeQtGraphicsView::setMazeObject(size_t row, size_t column, MazeObject obj) {
    this->maze->at(row).at(column)->setType(obj);
}

void MazeQtGraphicsView::clearRoadHit() {

    /* to traversal obj or maze */
    for (int row = 0; row < this->originMaze->getSize().first; row++)
    {
        for (int column = 0; column < this->originMaze->getSize().second; column++)
        {
            // if obj is road, set to blank
            if (this->maze->at(row).at(column)->getType() == RoadHint)
            {
                // set origin maze obj
                this->originMaze->at(row, column) = Blank;

                // set graphics view obj
                this->maze->at(row).at(column)->setType(Blank);
            }

        }
    }

}

