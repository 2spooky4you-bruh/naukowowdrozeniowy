from CNN import Recognition

network = Recognition()

SIZE_IMAGE = 128 #1
ADJUST_GAMMA = 1.0
GRABCUT = True ###
EQUALIZE_HIST = False
ROTATE = False
BRIGHTNESS = 1.0
BLUR = False
config_name ='Size=' +  str(SIZE_IMAGE) + 'Grabcut=' + str(GRABCUT) + 'Blur=' + str(BLUR) + 'Gamma=' + str(ADJUST_GAMMA) + 'EqHist=' + str(EQUALIZE_HIST) + 'Rotate=' + str(ROTATE) + 'Brightness=' + str(BRIGHTNESS)
#network.train_net(config_name, SIZE_IMAGE)

SIZE_IMAGE = 128 #2
ADJUST_GAMMA = 1.0
GRABCUT = False
EQUALIZE_HIST = False
ROTATE = False
BRIGHTNESS = 1.0
BLUR = False
config_name ='Size=' +  str(SIZE_IMAGE) + 'Grabcut=' + str(GRABCUT) + 'Blur=' + str(BLUR) + 'Gamma=' + str(ADJUST_GAMMA) + 'EqHist=' + str(EQUALIZE_HIST) + 'Rotate=' + str(ROTATE) + 'Brightness=' + str(BRIGHTNESS)
#network.train_net(config_name, SIZE_IMAGE)

SIZE_IMAGE = 128 #3
ADJUST_GAMMA = 1.0
GRABCUT = False
EQUALIZE_HIST = True ###
ROTATE = False
BRIGHTNESS = 1.0
BLUR = False
config_name ='Size=' +  str(SIZE_IMAGE) + 'Grabcut=' + str(GRABCUT) + 'Blur=' + str(BLUR) + 'Gamma=' + str(ADJUST_GAMMA) + 'EqHist=' + str(EQUALIZE_HIST) + 'Rotate=' + str(ROTATE) + 'Brightness=' + str(BRIGHTNESS)
#network.train_net(config_name, SIZE_IMAGE)

SIZE_IMAGE = 128 #4
ADJUST_GAMMA = 1.0
GRABCUT = False
EQUALIZE_HIST = False
ROTATE = True ###
BRIGHTNESS = 1.0
BLUR = False
config_name ='Size=' +  str(SIZE_IMAGE) + 'Grabcut=' + str(GRABCUT) + 'Blur=' + str(BLUR) + 'Gamma=' + str(ADJUST_GAMMA) + 'EqHist=' + str(EQUALIZE_HIST) + 'Rotate=' + str(ROTATE) + 'Brightness=' + str(BRIGHTNESS)
#network.train_net(config_name, SIZE_IMAGE)


SIZE_IMAGE = 128 #5
ADJUST_GAMMA = 1.0
GRABCUT = False
EQUALIZE_HIST = False
ROTATE = False
BRIGHTNESS = 1.0
BLUR = False
config_name ='Size=' +  str(SIZE_IMAGE) + 'Grabcut=' + str(GRABCUT) + 'Blur=' + str(BLUR) + 'Gamma=' + str(ADJUST_GAMMA) + 'EqHist=' + str(EQUALIZE_HIST) + 'Rotate=' + str(ROTATE) + 'Brightness=' + str(BRIGHTNESS)
#network.train_net(config_name, SIZE_IMAGE)

SIZE_IMAGE = 128 #6
ADJUST_GAMMA = 0.5 ###
GRABCUT = False
EQUALIZE_HIST = False
ROTATE = False
BRIGHTNESS = 1.0
BLUR = False
config_name ='Size=' +  str(SIZE_IMAGE) + 'Grabcut=' + str(GRABCUT) + 'Blur=' + str(BLUR) + 'Gamma=' + str(ADJUST_GAMMA) + 'EqHist=' + str(EQUALIZE_HIST) + 'Rotate=' + str(ROTATE) + 'Brightness=' + str(BRIGHTNESS)
#network.train_net(config_name, SIZE_IMAGE)

SIZE_IMAGE = 128 #7
ADJUST_GAMMA = 1.5 ###
GRABCUT = False
EQUALIZE_HIST = False
ROTATE = False
BRIGHTNESS = 1.0
BLUR = False
config_name ='Size=' +  str(SIZE_IMAGE) + 'Grabcut=' + str(GRABCUT) + 'Blur=' + str(BLUR) + 'Gamma=' + str(ADJUST_GAMMA) + 'EqHist=' + str(EQUALIZE_HIST) + 'Rotate=' + str(ROTATE) + 'Brightness=' + str(BRIGHTNESS)
#network.train_net(config_name, SIZE_IMAGE)

SIZE_IMAGE = 128 #8
ADJUST_GAMMA = 1.0
GRABCUT = False
EQUALIZE_HIST = False
ROTATE = False
BRIGHTNESS = 0.5 ###
BLUR = False
config_name ='Size=' +  str(SIZE_IMAGE) + 'Grabcut=' + str(GRABCUT) + 'Blur=' + str(BLUR) + 'Gamma=' + str(ADJUST_GAMMA) + 'EqHist=' + str(EQUALIZE_HIST) + 'Rotate=' + str(ROTATE) + 'Brightness=' + str(BRIGHTNESS)
#network.train_net(config_name, SIZE_IMAGE)

SIZE_IMAGE = 128 #10
ADJUST_GAMMA = 1.0
GRABCUT = False
EQUALIZE_HIST = False
ROTATE = False
BRIGHTNESS = 1.5 ###
BLUR = False
config_name ='Size=' +  str(SIZE_IMAGE) + 'Grabcut=' + str(GRABCUT) + 'Blur=' + str(BLUR) + 'Gamma=' + str(ADJUST_GAMMA) + 'EqHist=' + str(EQUALIZE_HIST) + 'Rotate=' + str(ROTATE) + 'Brightness=' + str(BRIGHTNESS)
#network.train_net(config_name, SIZE_IMAGE)

SIZE_IMAGE = 64 #11 ###
ADJUST_GAMMA = 1.0
GRABCUT = False
EQUALIZE_HIST = False
ROTATE = False
BRIGHTNESS = 1.0
BLUR = False
config_name ='Size=' +  str(SIZE_IMAGE) + 'Grabcut=' + str(GRABCUT) + 'Blur=' + str(BLUR) + 'Gamma=' + str(ADJUST_GAMMA) + 'EqHist=' + str(EQUALIZE_HIST) + 'Rotate=' + str(ROTATE) + 'Brightness=' + str(BRIGHTNESS)
#network.train_net(config_name, SIZE_IMAGE)

SIZE_IMAGE = 196 #12 ###
ADJUST_GAMMA = 1.0
GRABCUT = False
EQUALIZE_HIST = False
ROTATE = False
BRIGHTNESS = 1.0
BLUR = False
config_name ='Size=' +  str(SIZE_IMAGE) + 'Grabcut=' + str(GRABCUT) + 'Blur=' + str(BLUR) + 'Gamma=' + str(ADJUST_GAMMA) + 'EqHist=' + str(EQUALIZE_HIST) + 'Rotate=' + str(ROTATE) + 'Brightness=' + str(BRIGHTNESS)
#network.train_net(config_name, SIZE_IMAGE)

SIZE_IMAGE = 128 #13
ADJUST_GAMMA = 1.0
GRABCUT = False
EQUALIZE_HIST = False
ROTATE = False
BRIGHTNESS = 1.0
BLUR = True ###
config_name ='Size=' +  str(SIZE_IMAGE) + 'Grabcut=' + str(GRABCUT) + 'Blur=' + str(BLUR) + 'Gamma=' + str(ADJUST_GAMMA) + 'EqHist=' + str(EQUALIZE_HIST) + 'Rotate=' + str(ROTATE) + 'Brightness=' + str(BRIGHTNESS)
network.train_net(config_name, SIZE_IMAGE)

