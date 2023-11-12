//
// Created by idk on 2023/10/30.
//

#include "MazeQtGraphicsPixmapItem.h"
#include <QGraphicsScene>
#include <QPoint>
#include <QGraphicsSceneMouseEvent>
#include <QMouseEvent>

MazeQtGraphicsPixmapItem::MazeQtGraphicsPixmapItem() : QGraphicsPixmapItem() {

    /* create texture if not exist */
    if (wallPixmap == nullptr) {
        wallPixmap = std::make_shared<QPixmap>("./res/wall.png");
    }

    if (blankPixmap == nullptr) {
        blankPixmap = std::make_shared<QPixmap>("./res/blank.png");
    }

    if (playerPixmap == nullptr) {
        playerPixmap = std::make_shared<QPixmap>("./res/player.png");
    }
    if (roadHintPixmap == nullptr) {
        roadHintPixmap = std::make_shared<QPixmap>("./res/road_hit.png");
    }

    if (startPixmap == nullptr) {
        startPixmap = std::make_shared<QPixmap>("./res/start.png");
    }

    if (endPixmap == nullptr) {
        endPixmap = std::make_shared<QPixmap>("./res/end.png");
    }

    this->setType(MazeObject::Wall);
}

MazeQtGraphicsPixmapItem::~MazeQtGraphicsPixmapItem() = default;

void MazeQtGraphicsPixmapItem::setType(MazeObject obj) {

    /* change texture */
    switch (obj) {
        case Wall:
            this->setPixmap(*wallPixmap);
            break;

        case Blank:
            this->setPixmap(*blankPixmap);
            break;

        case Player:
            this->setPixmap(*playerPixmap);
            break;

        case RoadHint:
            this->setPixmap(*roadHintPixmap);
            break;
        case Start:
            this->setPixmap(*startPixmap);
            break;
        case End:
            this->setPixmap(*endPixmap);
            break;
    }

    /* change type */
    this->mazeObjectType = obj;
}

QSize MazeQtGraphicsPixmapItem::getSize() {
    return this->pixmap().size();
}

MazeObject MazeQtGraphicsPixmapItem::getType() {
    return this->mazeObjectType;
}

void MazeQtGraphicsPixmapItem::setIndex(size_t row, size_t column) {
    this->row = row;
    this->column = column;
}

std::pair<size_t, size_t> MazeQtGraphicsPixmapItem::getIndex() {

    // check is set index
    if (this->row == -1 && this->column == -1) {
        qDebug() << "(MazeQtGraphicsPixmapItem) Error : not set index ";
        exit(-1);
    }

    return std::pair<size_t, size_t>(row, column);
}



