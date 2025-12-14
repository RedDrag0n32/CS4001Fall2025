radio.onReceivedNumber(function on_received_number(receivedNumber: number) {
    
    fearFrequency = input.soundLevel()
})
let fearFrequency = 0
radio.setGroup(1)
fearFrequency = 0
basic.forever(function on_forever() {
    if (fearFrequency > 0 && fearFrequency <= 150) {
        basic.showIcon(IconNames.Sad)
        cuteBot.colorLight(cuteBot.RGBLights.RGB_L, 0xffff00)
        cuteBot.colorLight(cuteBot.RGBLights.RGB_R, 0xffff00)
        cuteBot.motors(100, -100)
        radio.sendNumber(input.soundLevel())
    } else if (fearFrequency >= 150) {
        music._playDefaultBackground(music.builtInPlayableMelody(Melodies.Funeral), music.PlaybackMode.InBackground)
        basic.showIcon(IconNames.Skull)
        cuteBot.colorLight(cuteBot.RGBLights.RGB_L, 0xff0000)
        cuteBot.colorLight(cuteBot.RGBLights.RGB_R, 0xff0000)
        for (let index = 0; index < 4; index++) {
            cuteBot.moveTime(cuteBot.Direction.forward, 100, 1)
            cuteBot.stopcar()
            basic.pause(100)
            cuteBot.motors(100, -100)
            basic.pause(100)
            cuteBot.motors(100, -100)
            basic.pause(100)
            cuteBot.moveTime(cuteBot.Direction.forward, 100, 1)
        }
        radio.sendNumber(input.soundLevel())
    } else {
        cuteBot.colorLight(cuteBot.RGBLights.RGB_L, 0x00ff00)
        cuteBot.colorLight(cuteBot.RGBLights.RGB_R, 0x00ff00)
        basic.showIcon(IconNames.Happy)
        radio.sendNumber(input.soundLevel())
    }
    
})