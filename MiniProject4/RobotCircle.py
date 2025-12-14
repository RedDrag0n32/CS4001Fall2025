frequency = 3.9
basic.show_leds("""
    . . . # .
    # . # . #
    # . # . #
    # . # . #
    . # . . .
    """)

def on_forever():
    if frequency >= 3.5:
        if cuteBot.tracking(cuteBot.TrackingState.L_UNLINE_R_LINE):
            cuteBot.motors(25, 10)
        if cuteBot.tracking(cuteBot.TrackingState.L_LINE_R_UNLINE):
            cuteBot.motors(10, 25)
        if cuteBot.tracking(cuteBot.TrackingState.L_R_LINE):
            cuteBot.motors(25, 25)
    else:
        cuteBot.move_time(cuteBot.Direction.FORWARD, 100, 3)
        cuteBot.move_time(cuteBot.Direction.BACKWARD, 25, 3)
basic.forever(on_forever) 