from pynput import mouse, keyboard
from time import sleep


class MyException(Exception): pass


def check_for_exit():
    with keyboard.Events as events:
        for event in events:
            print(event)
            if isinstance(event, keyboard.Key.esc):
                return True
    return False


def cycle_click(x, y, time):
    my_mouse = mouse.Controller()
    print(f'Запускаем кликер с периодичностью {time} минут')
    while True:
        my_mouse.position = (x, y)
        my_mouse.click(mouse.Button.left)
        print('Произведен клик!')
        sleep(time * 60)
        if check_for_exit():
            return False


def check_setting_click(x, y):
    answer = input(f'({x}, {y}) Это желаемые координаты? y/n\n')
    if answer == 'y':
        print(answer)
        time = float(input('Введите время в минутах (целое или дробное число)\n'))
        cycle_click(x, y, time)
    else:
        print('Ну, нажмите еще раз')
        main()


def main():
    print('Нажмите на кнопку Play')
    with mouse.Events() as events:
        for event in events:
            if isinstance(event, mouse.Events.Click) and event.button == mouse.Button.left:
                pos = mouse.Controller().position
                check_setting_click(pos[0], pos[1])
            else:
                # print('Received event {}'.format(event))
                ...


main()
