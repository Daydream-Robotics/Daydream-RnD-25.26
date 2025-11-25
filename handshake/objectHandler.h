#ifndef OBJECT_HANDLER
#define OBJECT_HANDLER
#include <array>
#include <optional>

enum GamePiece {
    RED_BALL,
    BLUE_BALL,
    LONG_GOAL,
    MID_GOAL_1,
    MID_GOAL_2,
    MID_GOAL_3,
    MID_GOAL_4,
    COUNT
};

typedef struct GamePieceData{
	int x;
	int y;
	float conf;
}GamePieceData;

GamePieceData getClosestObject(GamePiece gamePiece);

using GamePieceArray = std::array<std::optional<GamePieceData>,static_cast<size_t>(GamePiece::COUNT)>;

constexpr size_t RED_BALL_INDEX = static_cast<size_t>(GamePiece::RED_BALL);
constexpr size_t BLUE_BALL_INDEX = static_cast<size_t>(GamePiece::BLUE_BALL);
constexpr size_t LONG_GOAL_INDEX = static_cast<size_t>(GamePiece::LONG_GOAL);
constexpr size_t MID_GOAL_1_INDEX = static_cast<size_t>(GamePiece::MID_GOAL_1);
constexpr size_t MID_GOAL_2_INDEX = static_cast<size_t>(GamePiece::MID_GOAL_2);
constexpr size_t MID_GOAL_3_INDEX = static_cast<size_t>(GamePiece::MID_GOAL_3);
constexpr size_t MID_GOAL_4_INDEX = static_cast<size_t>(GamePiece::MID_GOAL_4);
#endif
