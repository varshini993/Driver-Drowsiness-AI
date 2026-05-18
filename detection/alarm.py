import pygame

alarm_playing = False
alarm_loaded = False

try:

    pygame.mixer.pre_init(44100, -16, 2, 512)
    pygame.init()

    alarm_sound = pygame.mixer.Sound("static/sounds/alarm.wav")

    alarm_loaded = True

    print("Alarm loaded successfully")

except Exception as e:

    print("Alarm loading failed:")
    print(e)


def play_alarm():

    global alarm_playing

    if alarm_loaded and not alarm_playing:

        alarm_sound.play(-1)

        alarm_playing = True


def stop_alarm():

    global alarm_playing

    if alarm_loaded and alarm_playing:

        alarm_sound.stop()

        alarm_playing = False