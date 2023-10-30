//
// Created by idk on 2023/10/30.
//

#include "MazeQtGraphicsView.h"
#include <QMouseEvent>
#include <QGraphicsView>
#include <QGraphicsScene>
#include <QGraphicsPixmapItem>
#include <QPixmap>



MazeQtGraphicsView::MazeQtGraphicsView(QWidget *parent) : QGraphicsView(parent) {

//
    MazeQtGraphicsPixmapItem *img = new MazeQtGraphicsPixmapItem();
    QPixmap qp;
    qp.load("/home/idk/Pictures/wall.png");
    img->setPixmap(qp);
    img->setPos(100, 100);

// Create a scene and add a text item to it.
    QGraphicsScene *scene = new QGraphicsScene(this);
    setScene(scene);
    QGraphicsTextItem *textItem = scene->addText("Click me!");

    scene->addItem(img);

    textItem->setPos(100, 100);

}


void MazeQtGraphicsView::mousePressEvent(QMouseEvent *event) {

    // Get the position of the click event.
    QPointF clickPos = mapToScene(event->pos());

    // Use scene()->items() to find items at the click position.
    QList<QGraphicsItem *> items = scene()->items(clickPos);

    for (QGraphicsItem *item : items)
    {
        if (item->type() == MazeQtGraphicsPixmapItem::Type)
        {
            // Change the content of the QGraphicsPixmapItem.
            auto *pixmapItem = dynamic_cast<MazeQtGraphicsPixmapItem *>(item);

            pixmapItem->setPos(1000, 1000);



        }
    }

    // Call the base class implementation to ensure normal event handling.
    QGraphicsView::mousePressEvent(event);
}