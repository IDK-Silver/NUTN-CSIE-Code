//
// Created by idk on 2023/10/30.
//

#ifndef MAZE_MAZE_QT_GRAPHICS_PIXMAP_ITEM_H
#define MAZE_MAZE_QT_GRAPHICS_PIXMAP_ITEM_H

#include <memory>
#include <QGraphicsPixmapItem>
#include "../Maze.h"

static std::shared_ptr<QPixmap> wallPixmap;
static std::shared_ptr<QPixmap> blankPixmap;
static std::shared_ptr<QPixmap> playerPixmap;
static std::shared_ptr<QPixmap> roadHintPixmap;
static std::shared_ptr<QPixmap> startPixmap;
static std::shared_ptr<QPixmap> endPixmap;

class MazeQtGraphicsPixmapItem : public QGraphicsPixmapItem {
public:



    MazeQtGraphicsPixmapItem();
    ~MazeQtGraphicsPixmapItem() override;

    /* the method to get and set MazeObject */
    void setType(MazeObject obj);
    MazeObject getType();

    /* to get maze size */
    QSize getSize();

    /* to set or get index */
    std::pair<size_t, size_t> getIndex();
    void setIndex(size_t row, size_t column);

private:



protected:
    /* maze type */
    MazeObject mazeObjectType;

    /* index */
    int  row = -1, column = -1;

};


#endif //MAZE_MAZE_QT_GRAPHICS_PIXMAP_ITEM_H
