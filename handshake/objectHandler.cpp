#include <cstring>//string manipulation AUTO
#include "objectHandler.h"
#include <iostream>

char firstchar = '0';

GamePieceArray init_array(){
    //return if any statement fails
    GamePieceArray failedinitarray;
    //size can contain 1 frame of all detected objects
    char inputbuffer[106];

    //begin serial handshake
    printf("A\n");
    fflush(stdout);

    if(fgets(inputbuffer,sizeof(inputbuffer),stdin) != NULL){
        firstchar = inputbuffer[0];
        GamePieceArray framearray;
        //temporarily store GamePieceData
        int classidparse;
        int xparse;
        int yparse;
        float confparse;

        //tokenize inputbuffer
        //class_id
		char* token = strtok(inputbuffer,",|\n");
		//store class_id
        if((sscanf(token,("%d"),&classidparse))!=1)
            return failedinitarray;

        //x_center
		token = strtok(NULL,",|\n");//115
		if((sscanf(token,("%d"),&xparse))!=1)
            return failedinitarray;
        //y_center
		token = strtok(NULL,",|\n");
		if((sscanf(token,("%d"),&yparse))!=1)
            return failedinitarray;
        //confidence value
		token = strtok(NULL,",|\n");
		if((sscanf(token,("%f"),&confparse))!=1)
            return failedinitarray;

        GamePieceData framedata={.x=xparse,.y=yparse,.conf=confparse};

		switch(classidparse){
            case RED_BALL:
                framearray[RED_BALL_INDEX]=framedata;
                break;
            case BLUE_BALL:
                framearray[BLUE_BALL_INDEX]=framedata;
                break;
            case LONG_GOAL:
                framearray[LONG_GOAL_INDEX]=framedata;
                break;
            case MID_GOAL_1:
                framearray[MID_GOAL_1_INDEX]=framedata;
                break;
            case MID_GOAL_2:
                framearray[MID_GOAL_2_INDEX]=framedata;
                break;
            case MID_GOAL_3:
                framearray[MID_GOAL_3_INDEX]=framedata;
                break;
            case MID_GOAL_4:
                framearray[MID_GOAL_4_INDEX]=framedata;
                break;
            default:
                return failedinitarray;
		}

		while(token != NULL){
            //next detected object
            //class_id
            if((token = strtok(NULL,",|\n"))==NULL)
                break;
            //store class_id
            if((sscanf(token,("%d"),&classidparse))!=1)
                return failedinitarray;
            //x_center
            token = strtok(NULL,",|\n");
            if((sscanf(token,("%d"),&xparse))!=1)
                return failedinitarray;
            //y_center
            token = strtok(NULL,",|\n");
            if((sscanf(token,("%d"),&yparse))!=1)
                return failedinitarray;
            //confidence value
            token = strtok(NULL,",|\n");
            if((sscanf(token,("%f"),&confparse))!=1)
                return failedinitarray;

            framedata={.x=xparse,.y=yparse,.conf=confparse};

            switch(classidparse){
                case RED_BALL:
                    framearray[RED_BALL_INDEX]=framedata;
                    break;
                case BLUE_BALL:
                    framearray[BLUE_BALL_INDEX]=framedata;
                    break;
                case LONG_GOAL:
                    framearray[LONG_GOAL_INDEX]=framedata;
                    break;
                case MID_GOAL_1:
                    framearray[MID_GOAL_1_INDEX]=framedata;
                    break;
                case MID_GOAL_2:
                    framearray[MID_GOAL_2_INDEX]=framedata;
                    break;
                case MID_GOAL_3:
                    framearray[MID_GOAL_3_INDEX]=framedata;
                    break;
                case MID_GOAL_4:
                    framearray[MID_GOAL_4_INDEX]=framedata;
                    break;
                default:
                    return failedinitarray;
            }
        }//end of while
        return framearray;
    }else{
        return failedinitarray;
    }
}

/*returns an array of std::optional elements of enum GamePiece class COUNT size.
each std::optional element wraps a GamePieceData structure.
see cpprefernce std::optional webpage for member functions.
*/
//may be used for pre-processing the array before it
//is returned to caller
GamePieceArray get_obj(){
	return init_array();
}

int main(void){
    while(firstchar != '1'){

        GamePieceArray giveframearray = get_obj();

        if(giveframearray[RED_BALL_INDEX].has_value()){
            printf("\ndetected red ball\nx:\t%d\ny:\t%d\nconfidence value:\t%f\n",
               (giveframearray[RED_BALL_INDEX].value()).x,
               (giveframearray[RED_BALL_INDEX].value()).y,
               (giveframearray[RED_BALL_INDEX].value()).conf);
        }else
            printf("\nno red ball found\n");
        if(giveframearray[BLUE_BALL_INDEX].has_value()){
            printf("\ndetected blue ball\nx:\t%d\ny:\t%d\nconfidence value:\t%f\n",
               (giveframearray[BLUE_BALL_INDEX].value()).x,
               (giveframearray[BLUE_BALL_INDEX].value()).y,
               (giveframearray[BLUE_BALL_INDEX].value()).conf);
        }else
            printf("\nno blue ball found\n");
        if(giveframearray[LONG_GOAL_INDEX].has_value()){
            printf("\ndetected long goal\nx:\t%d\ny:\t%d\nconfidence value:\t%f\n",
               (giveframearray[LONG_GOAL_INDEX].value()).x,
               (giveframearray[LONG_GOAL_INDEX].value()).y,
               (giveframearray[LONG_GOAL_INDEX].value()).conf);
        }else
            printf("\nno long goal found\n");
        if(giveframearray[MID_GOAL_1_INDEX].has_value()){
            printf("\ndetected mid goal 1\nx:\t%d\ny:\t%d\nconfidence value:\t%f\n",
               (giveframearray[MID_GOAL_1_INDEX].value()).x,
               (giveframearray[MID_GOAL_1_INDEX].value()).y,
               (giveframearray[MID_GOAL_1_INDEX].value()).conf);
        }else
            printf("\nno mid goal 1 found\n");
        if(giveframearray[MID_GOAL_2_INDEX].has_value()){
            printf("\ndetected mid goal 2\nx:\t%d\ny:\t%d\nconfidence value:\t%f\n",
               (giveframearray[MID_GOAL_2_INDEX].value()).x,
               (giveframearray[MID_GOAL_2_INDEX].value()).y,
               (giveframearray[MID_GOAL_2_INDEX].value()).conf);
        }else
            printf("\nno mid goal 2 found\n");
        if(giveframearray[MID_GOAL_3_INDEX].has_value()){
            printf("\ndetected mid goal 3\nx:\t%d\ny:\t%d\nconfidence value:\t%f\n",
               (giveframearray[MID_GOAL_3_INDEX].value()).x,
               (giveframearray[MID_GOAL_3_INDEX].value()).y,
               (giveframearray[MID_GOAL_3_INDEX].value()).conf);
        }else
            printf("\nno mid goal 3 found\n");
        if(giveframearray[MID_GOAL_4_INDEX].has_value()){
            printf("\ndetected mid goal 4\nx:\t%d\ny:\t%d\nconfidence value:\t%f\n",
               (giveframearray[MID_GOAL_4_INDEX].value()).x,
               (giveframearray[MID_GOAL_4_INDEX].value()).y,
               (giveframearray[MID_GOAL_4_INDEX].value()).conf);
        }else
            printf("\nno mid goal 4 found\n");
    }

    return 0;
}//end of main() function

//0,150,200,0.95|1,320,100,0.88|2,480,350,0.79|3,50,450,0.92|4,600,50,0.85|5,400,250,0.72|6,250,50,0.98\n
//3,98,412,0.81|2,550,15,0.73|1,121,580,0.99|0,390,260,0.65|4,50,50,0.91|5,233,300,0.84|6,610,5,0.76\n
//3,500,105,0.77|6,145,390,0.82|1,288,250,0.91|0,55,15,0.96|5,405,450,0.68|2,200,88,0.73|4,330,330,0.89\n
