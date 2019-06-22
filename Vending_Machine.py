av = list(range(1,16))

    for i in av:
        x = (int(input('\nHow many candies do you want?\n'))) - 1
        if len(av) == 0:
         print('Sorry!! Out of stock, please try again later')
        y = x+1
        if x > len(av):
            print('Sorry!! Out of stock, please try again later')
            print('Thank you for using me. Please come again')
            break
        if x+1 > len(av):
            print('Sorry!! Not able dispense that many\nPlease decide on a lesser count')
            continue
        for i in av:
        if av.index(i) <= x:
            print('Candy,', end=" ")
            if av.index(i) == x:
                print('\nThank you for using me. Come again. Bye!!')
                break
        if av.index(i) <= x:
            continue
    del av[len(av)-y:]
