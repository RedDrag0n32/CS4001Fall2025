emotion = "boring"


from microbit import *

def flash_to_microbit(game_state):
    if(game_state == "boring"):
        display.show(Image.MEH)

    elif(game_state == "calm"):
        display.show(Image.ASLEEP)
    
    elif(game_state == "horror"):
        display.show(Image.SKULL)
    
    else:
        display.show(Image.SMILE)


flash_to_microbit(emotion)




