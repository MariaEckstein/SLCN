# Lukas Nagel lukas@bccn-berlin.de
#
#
#
from direct.showbase.ShowBase import ShowBase
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import *
from direct.gui.OnscreenImage import OnscreenImage
from direct.showbase.Transitions import Transitions
from panda3d.core import TransparencyAttrib
from direct.actor.Actor import Actor
import random, sys, os, math
from direct.gui.OnscreenText import OnscreenText 
from direct.gui.DirectGui import *
from numpy import *
from direct.task import Task
import time

# if you have pygame than install joystick 
try:
    import pygame
    from pygame import joystick
    pygame.init()
    joystick.init()

except:
    print "Error: pygame not installed"



# making 2 classes - event manager and run 
def main():
    em = EventManager()
    em.run()
        
class EventManager(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)
        #ppyself.setFrameRateMeter(True)
        self.agui()
        self.initkeys()
        self.game_init = False
        self.rungame = False
        taskMgr.add(self.game, "Game",)
	# setting up the beginning settings 
    def agui(self):
        self.a,self.b = text_entry("Particpant ID",0.6)
        self.c,self.d = text_entry("Age",0.5)
        self.e,self.f = text_entry("Gender",0.4,options = ["Male","Female"])
        self.butt = button("Go!",-0.5,extras=[self.b,self.d,self])

        self.p_ID = self.b.get()
        self.todelete = [self.c,self.d,self.a,self.b,self.e,self.f,self.butt]

    def game(self,task):
        if self.game_init:
            self.initgame()
            for thing in self.todelete:
                thing.destroy()
            self.game_init = False
        #if you have started the game 
        if self.rungame:
        	# and as long as the game isn't over 
            if self.GameOver == False:
            	# if you are on a new trial 
                if self.new_trial:
                	# then start a new trial 
                    self.init_trial()
                # if you just finished a trial
                # then show the "ready set go" display
                elif self.go:
                    self.godisp()
                # if you have the joystick then get info from that
                else:
                    if self.joystickon == 1:
                        self.getjoystick()
                    #otherwise, just use the regular move functions 
                    else:
                        self.move()
                    self.animate()
                    self.dig()
                    self.drag()
                self.movecam()
                if self.savelog:
                    self.log()
                    self.savelog = False
            else: # the game over view 
                if self.BlackScreen == 0:
                    self.BlackScreen = 1
                    OnscreenImage(image = 'textures/black.jpg', pos = (0, 0, 0),scale=100)
                    self.starimage = OnscreenImage(image = 'textures/goldstar.png', pos = (0,0,0),scale=0.1)
                    self.starimage.setTransparency(TransparencyAttrib.MAlpha)
                  #  self.EndText = OnscreenText(text = "GREAT JOB!" , pos = (-100, -0.5), scale = 0.1*self.goscale)
        return task.cont
        
	# recording all of the info to save for the log
    def log(self):
        """Save trial
        ID - Age - Gender - Reversal - 
                        """
        self.file.write("\n")
        self.file.write("%s\t" % self.p_ID)
        self.file.write("%s\t" % self.p_age)
        self.file.write("%s\t" % self.p_gender)
        self.file.write("%s\t" % self.trial_no)
        self.file.write("%s\t" % self.reversal)
        self.file.write("%s\t" % self.trial[0])
        self.file.write("%s\t" % self.dug)
        self.file.write("%s\t" % self.trialscore)
        self.file.write("%s\t" % self.TotalScore)
        self.file.write("%s\t" % self.RunningTotal)
        for visit_no in self.visits:
            self.file.write("%s\t" % visit_no)
        self.file.write("%s\t" % self.RT)

	# showing the ready set go display 
    def godisp(self):
        self.ReadyText.destroy()
        self.GoText.destroy()
        self.goscale+=60./self.framerate
        if self.goscale < 60:
            self.ReadyText = OnscreenText(text = "Ready!" , pos = (0, -0.5), scale = 0.1*self.goscale/2)
        elif self.goscale < 120:
            self.GoText.destroy()
        elif self.goscale < 180:
            self.GoText = OnscreenText(text = "Go!" , pos = (0, -0.5), scale = 0.1*(self.goscale-120)/2)
        else: 
            self.go = False
            self.t0 = time.time()

	# how to move with joystick
    def getjoystick(self):
        dt = globalClock.getDt()
        if self.canMove == True:
            for event in pygame.event.get():
                pass
            self.isMoving = False
            self.isTurning = False
            if self.joystick.get_axis( 4 )<-0.8:
                self.isMoving=True
                self.get_move(dt)
            if self.joystick.get_axis( 3 )<-0.8:
                self.kiki.setH(self.kiki.getH() + 300 * dt)
                self.isTurning = True
            if self.joystick.get_axis( 3 )>0.8:
                self.kiki.setH(self.kiki.getH() - 300 * dt)
                self.isTurning = True
        if self.joystick.get_button( 1 ) > 0.8:
            if self.get_dig()>-1:
                self.digging = True



	# how to move without a joystick 
    def move(self):
        dt = globalClock.getDt() #get time from last frame
        #check for keyboard input
        if self.canMove == True:
            if self.keyMap["spin"]:
                pass
                #self.camspin = True
            else:
                self.camspin = False
            if self.keyMap["left"]:
                self.kiki.setH(self.kiki.getH() + 100* dt)
                self.isTurning = True
            if self.keyMap["right"]:
                self.kiki.setH(self.kiki.getH() - 100* dt)
                self.isTurning = True
            if self.keyMap["forward"]:
                self.isMoving=True
                self.get_move(dt)
            else: self.isMoving = False
            if not self.keyMap["left"] and not self.keyMap["right"]:
                self.isTurning = False
        if self.keyMap["dig"]:
            if self.get_dig()>-1:
                self.digging = True
	# how to dig and what if win star
    def get_dig(self):
        x,y = self.kiki.getPos()[0],self.kiki.getPos()[1]
        dig = -1
        for i,j in zip(self.presLoc,[0,1,2,3]):
            if abs(x -self.boxloca[i][0])<3 and abs(y -self.boxloca[i][1])<3:
                dig = j
                if self.dug == -1:
                    self.RT = time.time() - self.t0 
                    self.savelog = True 
                self.dug = j
                if self.dug == self.trial[0]:
                    if self.trialscore == 0:
                        self.TotalScore+=1
                        self.addstar = 1
                        self.trialscore = 1
                        self.RunningTotal+=1
                else: 
                    self.RunningTotal = 0
        return dig

    def dig(self):
        if self.digging == True:
            self.plnp.setPos(self.boxloca[self.presLoc[self.dug]][0],self.boxloca[self.presLoc[self.dug]][1], 0)
            self.render.setLight(self.plnp)
            self.isMoving = False
            self.canMove=False
            if self.boxcam == True:
            		# if you found the star, spin! 
                    if self.dug == self.trial[0]:
                        self.star.reparentTo(self.render)
                        self.starspin()
                    # if you didn't find star, negative feedback cloud 
                    elif self.negfeedbac ==1:
                        if self.clouddisp == 1:
                            self.clouddisp =0
                            ex = 0
                            self.presents[self.dug].detachNode()
                            self.clouds.reparentTo(self.render)
                            self.clouds.setPos(self.boxloca[self.presLoc[self.dug]][0],self.boxloca[self.presLoc[self.dug]][1],2)
                        # if you 
                        if self.dragback == False:
                            if flipbook_2(self.clouds,self.cloudsframe,loop=0):
                                self.dragback= True
                                self.clouds.detachNode()
                                self.clouds.currentframe = 0
                                self.digging = False
                            # if lost, drag back to center 
                            else:
                	           flipbook_2(self.kiki,self.losingframe,loop=1)
                	# stop digging and drag back 
                    else:
                        self.digging = False
                        self.dragback= True
            else:
                a = flipbook_2(self.kiki,self.digframe,0)
                flipbook_2(self.presents[self.dug],self.presentsframe,0)
                if a:
                    self.boxcam = True
	# spinning if you find the star
    def starspin(self):
        self.presents[self.dug].detachNode()
        self.spin +=60./self.framerate
        # spin until reach 180 degrees 
        if self.spin < 180:
            self.star.setH(self.spin*5)
            self.star.setPos(self.boxloca[self.presLoc[self.dug]][0],self.boxloca[self.presLoc[self.dug]][1],3+self.spin/30.)
            flipbook_2(self.kiki,self.winframe,loop=1)
        # once you finish spin, then get dragged back 
        else: 
            self.dragback = True
            self.star.detachNode()
            self.digging = False
	# how to actually track moving
    
    def get_move(self,dt):
        dir = -self.kiki.getH()/360*2*pi
        prop = [self.kiki.getPos()[0]-25*sin(dir)*dt,self.kiki.getPos()[1]-25*cos(dir)*dt]
        pos = self.kiki.getPos()
        i,j = prop
        if prop[0]<-13:
            i=-13
        if prop[0]>13:
            i = 13
        if prop[1]>13:
            j = 13
        if prop[1]<-13:
            j = -13
        if pos[1]>=-10 and prop[1]<-10 and (prop[0]<-10 or prop[0]>10):
            j=-10
        if pos[1]<=10 and prop[1]>10 and (prop[0]<-10 or prop[0]>10):
            j=10
        if pos[0]>=-10 and prop[0]<-10 and (prop[1]<-10 or prop[1]>10):
            i=-10
        if pos[0]<=10 and prop[0]>10 and (prop[1]<-10 or prop[1]>10):
            i=10
        self.kiki.setPos(i,j,self.kiki.getPos()[2])
        self.check_location()
	# checking current position 
    def check_location(self):
        pos = self.kiki.getPos()

        if pos[0]<=-10+self.change_dist and pos[1]<=-10+self.change_dist:
            a = 0
        elif pos[0]>=10-self.change_dist and pos[1]<=-10+self.change_dist:
            a = 1
        elif pos[0]>=10-self.change_dist and pos[1]>=10-self.change_dist:
            a = 2
        elif pos[0]<=-10+self.change_dist and pos[1]>=10-self.change_dist:
            a = 3
        else: # if you can see a box, it's grey 
            if self.atbox > -1:
                self.presents[self.atbox].setTexture(self.texgrey)
                self.atbox = -1
            a = -1

		# if you get closer to the box, then it turns color (texture) 
        if a > -1:
            if self.atbox == -1:
                self.visits[self.presLoc.index(a)]+=1
            self.atbox = self.presLoc.index(a)
            self.presents[self.atbox].setTexture(self.textures[self.atbox])
            
        
	# setting animation of kiki to look like she's moving 
    def animate(self):
        if self.isMoving:
            flipbook_2(self.kiki,self.runframe)
        else: self.kiki.pose("go", 0)
	# how to actually drag kiki back to center 
    def drag(self):
        if self.dragback:
            if self.addstar == 1:
                self.starimage = OnscreenImage(image = 'textures/goldstar.png', pos = (-1.2+self.starLoc[self.TotalScore-1][0]*0.2, -0.8, -0.8+0.1*self.starLoc[self.TotalScore-1][1]),scale=0.1)
                self.starimage.setTransparency(TransparencyAttrib.MAlpha)
                self.addstar = -1
            self.floater.setZ(1.0)
            self.dragtime+=60./self.framerate
            if self.dragtime < self.time_out:
                pos = self.kiki.getPos()
                step = [-pos[0]/(self.time_out-self.dragtime),-pos[1]/(self.time_out-self.dragtime)]
                self.kiki.setPos(pos[0]+step[0],pos[1]+step[1],1)
                flipbook_2(self.kiki,self.dragframe,loop=1,)
            else: 
                self.new_trial = True

    def init_trial(self):
    	# if meet criteria for reversal, reverse! 
        if self.RunningTotal == self.revafter and self.reversal == 0:
            self.reversal = 1
            self.RunningTotal = 0
            # change correct box to a different box
            
            self.correctbox = 2
            #([i for i in range(3) if i != self.correctbox])
            self.textures[-1] = self.tex5
            # if meet criteria in reversal phase, end game! 
        elif self.RunningTotal == self.revafter and self.reversal == 1:
            self.GameOver = True
        #otherwise, continue for trials in acquisition
        self.camspinning = 0
        self.camspin = False
        self.render.setLightOff(self.plnp)
        self.new_trial = False
        self.boxcam = False
        self.go = True
        self.goscale = 0
        self.ReadyText = OnscreenText(text = "Ready..." , pos = (-100, -0.5), scale = 0.1*self.goscale)
        self.GoText = OnscreenText(text = "Go!" , pos = (-100, -0.5), scale = 0.1*self.goscale)
        self.floater.setZ(2.0)
        self.spin = 0
        self.dragback = False
        self.dragtime = 0
        self.trial_no += 1
        self.trialscore = 0
        self.trial = [self.correctbox]
        self.pos = [0.,0.]
        self.kiki.setPos(0,0,1.)
        self.kiki.setH(0)
        self.presLoc = [0,1,2,3]
        self.canMove = True
        self.digging = False
        self.atbox = -1
        self.addstar = -1
        self.clouddisp = 1
        ##### setting box locations
        random.shuffle(self.presLoc)
        self.boxloca = [[-12.,-12.],[12.,-12.],[12.,12.],[-12.,12.]]
        if self.trial_no>0:
            self.presents[self.dug].reparentTo(self.render)
        self.dug = -1
        for pres,loc in zip(self.presents,self.presLoc):
            pres.setPos(self.boxloca[loc][0],self.boxloca[loc][1],2.1)
            pres.setTexture(self.texgrey)
            if self.boxloca[loc][1] == -12:
                pres.setH(180)
            else: pres.setH(0)
            pres.currentframe = 0
            pres.pose("go",0)
        self.visits = [0,0,0,0]

	# setting up key controls 
    def initkeys(self):
        self.keyMap = {"left": 0, "right": 0, "forward": 0, "dig":0,"spin":0}
        self.accept("escape", sys.exit)
        self.accept("arrow_left", self.setKey, ["left", True])
        self.accept("arrow_right", self.setKey, ["right", True])
        self.accept("arrow_up", self.setKey, ["forward", True])
      # self.accept("arrow_done", self.setKey, ["backward", True])
        self.accept("d",self.setKey, ["dig",True])
        self.accept("y",self.setKey, ["spin",True])
        self.accept("arrow_left-up", self.setKey, ["left", False])
        self.accept("arrow_right-up", self.setKey, ["right", False])
        self.accept("arrow_up-up", self.setKey, ["forward", False])
        self.accept("d-up",self.setKey, ["dig",False])
        self.accept("y-up",self.setKey, ["spin",False])

	# take the key entered and make it the value 
    def setKey(self, key, value):
        self.keyMap[key] = value

	# initiate the game with settings of ID, age, gender
    def initgame(self):
        self.transition = Transitions(loader)
        self.getparameters()
        
        self.GameOver = False
        self.BlackScreen = 0

        self.p_ID = self.b.get()
        self.p_age = self.d.get()
        self.p_gender = self.f.get()
		# set the file saving the data with relevant variables 
        self.file = open('logs/Reversal3D_logfile_'+str(self.p_ID)+'.txt', 'w')
        self.file.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t" % 
            ("pID","Age","Gender","TrialNumber","Reversal","Target",
            "Choice","Correct","TotalScore","RunningTotal","Visits0","Visits1",
            "Visits2","Visits3","ResponseTime"))
		
        print "ID:", self.p_ID, "Gender:",self.p_gender,"Age:",self.p_age

		# running game to start
        self.rungame = True
        self.TotalScore = 0
        self.RunningTotal = 0
        self.trial_no = -1
        self.new_trial = True
        # setting the character kiki to move 
        self.kiki = CreateActor("Models/kiki.x","textures/kiki.jpg",0,0,0,0)
        self.kiki.currentframe= 0
        self.kiki.reparentTo(self.render)
        # starting without turning, ect. 
        self.isTurning = False
        self.lag = 0
        self.ScoreText = None
        
        self.framerate = 30
        globalClock.setMode(ClockObject.MLimited)
        globalClock.setFrameRate(self.framerate)
        self.frames = [int(round(i)) for i in arange(0,1200,60./self.framerate)]

		# making the boxes and the list of all 4 boxes 
        self.present1 = CreateActor("Models/test.x","textures/presentgrey.png",0,0,0,0)
        self.present1.reparentTo(self.render)
        self.present2 = CreateActor("Models/test.x","textures/presentgrey.png",0,0,0,0)
        self.present2.reparentTo(self.render)
        self.present3 = CreateActor("Models/test.x","textures/presentgrey.png",0,0,0,0)
        self.present3.reparentTo(self.render)
        self.present4 = CreateActor("Models/test.x","textures/presentgrey.png",0,0,0,0)
        self.present4.reparentTo(self.render)
        self.presents = [self.present1,self.present2,self.present3,self.present4]
        
        self.texgrey = loader.loadTexture("textures/presentgrey.png")
        
        # set the textures so that when kiki approaches, color changes 
        # there are 2 sets of box colors - A and B (counterbalance!) 
        if self.boxcol == "A":
            self.tex1 = loader.loadTexture("textures/presentblue.png")
            self.tex2 = loader.loadTexture("textures/presentyellow.jpg")
            self.tex3 = loader.loadTexture("textures/presentred.png")
            self.tex4 = loader.loadTexture("textures/presentpurple.png")
            self.tex5 = loader.loadTexture("textures/presentpink.jpg")
        else:
            self.tex1 = loader.loadTexture("textures/presentbrown.jpg")
            self.tex2 = loader.loadTexture("textures/presentorange.png")
            self.tex3 = loader.loadTexture("textures/presentgreen.png")
            self.tex4 = loader.loadTexture("textures/presentgreys.png")
            self.tex5 = loader.loadTexture("textures/presentlightpurple.png")
        
        # making the list of textures based on A or B 
        self.textures = [self.tex1,self.tex2,self.tex3,self.tex4]
        
        # setting the star - can change the star size 
        self.star = self.loader.loadModel("Models/star.x")
        tex = loader.loadTexture("textures/gold_texture.jpg")
        self.star.setTexture(tex)
        self.star.setScale(self.starsize)
        self.star.setPos(0, 0, 3)
        # setting the black cloud for negative feedback 
        self.clouds = CreateActor("Models/clouds.x","textures/cotton.jpg",0,0,0,0)
        self.star.setScale(self.cloudsize)
        # setting the room 
        self.environ = self.loader.loadModel("Models/Room.x")
        self.environ.reparentTo(self.render)
        # setting the wall color 
        # also 2 different wall colors A and B (counterbalance!) 
        if self.wallcol == "A":
            tex = loader.loadTexture("textures/arena.png")
        else:
            tex = loader.loadTexture("textures/room2.png")
        self.environ.setTexture(tex)
        self.environ.setScale(2, 2, 2)
        self.environ.setPos(0, 0, 0)

		# setting the start location 
        self.starLoc = []
        for i in range(4):
            for j in range(10):
                self.starLoc.append([i,j])
        # keeping track of location?
        self.floater = NodePath(PandaNode("floater"))
        self.floater.reparentTo(self.kiki)
        self.floater.setZ(2.0)
        
        self.angle = -2*pi*self.kiki.getH()/360
        
        alight = AmbientLight('alight')
        alight.setColor(VBase4(2, 2, 2, 1))
        alnp = render.attachNewNode(alight)
        render.setLight(alnp)
        self.plight = PointLight('plight')
        self.plight.setColor(VBase4(0, 0, 0, 0))
        self.plnp = render.attachNewNode(self.plight)
        
        #animation frames:
        self.runframe = get_frames(1,40,self.framerate)
        self.digframe = get_frames(60,240,self.framerate)
        self.presentsframe = get_frames(0,180,self.framerate)
        self.winframe = get_frames(320,360,self.framerate)
        self.losingframe = get_frames(380,460,self.framerate)
        self.dragframe = get_frames(242,302,self.framerate)
        self.cloudsframe = get_frames(0,180,self.framerate)
		# setting the correct box for acquisition phase 
        #self.correctbox = random.randint(0,3)
        self.correctbox = 0 
        self.reversal = 0
		# don't save the log until the end, when this becomes true 
        self.savelog = False
		# this isn't turned on unless you have pygame 
        self.joystickon = 0

        try:
            if joystick.get_count() > 0:
                self.joystickon = 1
                self.joystick = joystick.Joystick(0)
                self.joystick.init()
        except:
            print "Joystick Error" 
            self.joystickon = 0

	# these are the parameters available in menu 
	# menu take whatever you enter and puts it in the parameters.txt file 
    def getparameters(self):
        with open("parameters.txt",'r') as f:
            params= [x.strip().split('\t') for x in f]
        self.viewheight = int(params[1][1])
        self.change_dist = float(params[3][1])
        self.time_out = float(params[4][1])
        self.revafter = int(params[2][1])

        self.negfeedbac = {"Smoke-puff":1,"None":0}[params[5][1]]

        sizes = {"Small":0.5, "Medium":1, "Large":2}
        self.cloudsize = sizes[params[7][1]]
        self.starsize = sizes[params[6][1]]
        self.boxcol = params[8][1]
        self.wallcol = params[9][1]


	# this is the setting for moving around 
    def movecam(self):
        if self.camspin:
            self.camspinning += 1./10
            self.camera.setPos(self.kiki.getX()+(10*sin(self.angle+self.camspinning)), self.kiki.getY() + 10*cos(self.angle+self.camspinning), self.viewheight)
            self.camera.lookAt(self.floater)
        else:
            if self.isTurning:
            	# changed from 1
                self.lag=0.001
            elif self.lag > 0:
            	# changed from 1
                self.lag-=(0.001)*(6000/self.framerate)
            else: self.lag = 0
		# instead of -2
            newangle = 0*pi*self.kiki.getH()/360
            self.angle = newangle+(self.angle-newangle)*self.lag
            self.camera.setPos(self.kiki.getX()+(10*sin(self.angle)), self.kiki.getY() + 10*cos(self.angle),self.viewheight)
            i = self.camera.getPos()[0]
            j = self.camera.getPos()[1]

            if self.viewheight < 39:
                if self.camera.getPos()[0]>13:
                    i = 13
                if self.camera.getPos()[0]<-13:
                    i = -13
                if self.camera.getPos()[1]>13:
                    j = 13
                if self.camera.getPos()[1]<-13:
                    j = -13
            self.camera.setPos(i,j,self.camera.getPos()[2])
            if self.boxcam == False:
                self.camera.lookAt(self.floater)
            else:
                self.camera.setPos(self.kiki.getX()+(10*sin(self.angle)), self.kiki.getY() + 10*cos(self.angle), self.viewheight)
                i = self.camera.getPos()[0]
                j = self.camera.getPos()[1]
                if self.camera.getPos()[0]>13:
                    i = 13
                if self.camera.getPos()[0]<-13:
                    i = -13
                if self.camera.getPos()[1]>13:
                    j = 13
                if self.camera.getPos()[1]<-13:
                    j = -13
                self.camera.setPos(i,j,self.camera.getPos()[2])
                self.floater.setZ(2+self.spin/300.)
                self.camera.lookAt(self.floater)

    def seconds(self,secs):
        return secs*self.framerate
# where your're actually pulling the actor from
# this comes from one of the characters in the folder 
def CreateActor(fileName,texture,x,y,z,poseframe,):
    actor = Actor(fileName,{"go":fileName})
    tex = loader.loadTexture(texture)
    actor.setTexture(tex)
    actor.setPos(x,y,z)
    actor.pose("go", poseframe)
    actor.currentframe=poseframe
    return actor

def get_frames(start,end,framerate):
    return [int(round(i)) for i in arange(start,end,60./framerate)]

# moving the character with each frame 
def flipbook(actor,start,end,loop=1):
    if actor.currentframe<start:
        actor.currentframe=start
    if actor.currentframe<end:
       actor.currentframe+=1
    elif loop==1: 
        actor.currentframe = start
    else:
        actor.currentframe = start
        return 1
    actor.pose("go", actor.currentframe )

def flipbook_2(actor,frames,loop=1):
    if actor.currentframe<len(frames)-1:
       actor.currentframe+=1
    elif loop==1: 
        actor.currentframe = 0
    else:
        #actor.curentframe = 0
        return 1
    actor.pose("go", frames[actor.currentframe])
    return 0


######
def setText(textEntered):
    pass
 
#clear the text
def clearText():
    b.enterText('')

def text_entry(name,y,ax=-0.7,bx = 0, init="",options = []):
    a = OnscreenText(text = name, pos = (ax,y),
            scale = 0.07,fg=(1,0.5,0.5,1),align=TextNode.ALeft,mayChange=1)
    if len(options)<1:
        b = DirectEntry(text = "" ,scale=.06,pos = LVecBase3f(-bx,0,y),command=setText,
                initialText=init, numLines = 1,focus=0)
    else:
        b = DirectOptionMenu(text=options[0], scale=0.07,pos = LVecBase3f(-bx,0,y),items=options,initialitem=0,
                highlightColor=(1,0.5,0.5,1))

    return a,b

def check_vars(a,b,thingo):
    if len(a.get())>0 and len(a.get())>0:
        thingo.game_init = True
    else: return 0

# allowing you to accept button input? 
def button(name,y,x=0,extras=[]):
    b = DirectButton(text = name, scale=.1, command=check_vars,extraArgs = extras)
    return b

if __name__ == "__main__":
    main()
