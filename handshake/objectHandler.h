#ifndef OBJECT_HANDLER
#define OBJECT_HANDLER

enum GamePiece {
    RED_BALL,
    BLUE_BALL,
    MID_GOAL,
    LONG_GOAL
};

typedef struct GamePieceData{
	int x;
	int y;
	int conf;
}GamePieceData;

GamePieceData getClosestObject(GamePiece gamePiece);

#endif