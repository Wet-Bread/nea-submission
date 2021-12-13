import coz_commands
import pycozmo

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    with pycozmo.connect() as cli:
        cli.set_head_angle(0.3)

        while True:
            inp = input('You: ').lower()

            if inp == 'find face':
                coz_commands.find_face(cli)

            elif inp.split(' ')[0] == 'say':
                coz_commands.say(cli, inp[4:])

            elif inp == 'repeat':
                coz_commands.repeat(cli)

            elif inp == 'find body':
                coz_commands.find_body(cli)