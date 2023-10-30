//
// Created by idk on 2023/10/30.
//

#ifndef MAZE_MAZE_QT_GRAPHICS_PIXMAP_ITEM_H
#define MAZE_MAZE_QT_GRAPHICS_PIXMAP_ITEM_H

#include <QGraphicsPixmapItem>

class MazeQtGraphicsPixmapItem : public QGraphicsPixmapItem {
public:
    MazeQtGraphicsPixmapItem();
    ~MazeQtGraphicsPixmapItem();

    void setImage()

private:

protected:
    void mousePressEvent(QGraphicsSceneMouseEvent *mouseEvent) override;
};


#endif //MAZE_MAZE_QT_GRAPHICS_PIXMAP_ITEM_H
