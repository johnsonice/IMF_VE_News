import argparse

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--frequency_path', action='store', dest='frequency_path', default="NO NAME GIVEN")
    args = parser.parse_args()
    text_out = 'Passed freq path was {}'.format(args.frequency_path)

    f_out = open('demo.txt', 'w')
    f_out.write(text_out)
    f_out.close()
