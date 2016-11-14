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
import sys, os, math
from direct.gui.OnscreenText import OnscreenText 
from direct.gui.DirectGui import *
from numpy import *
import numpy as np
from direct.task import Task
import time

try:
    import pygame
    from pygame import joystick
    pygame.init()
    joystick.init()

except:
    print "Error: pygame not installed"




def main():
    em = EventManager()
    em.run()
        
class EventManager(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)
        #self.setFrameRateMeter(True)
        self.agui()
        self.initkeys()
        self.game_init = False
        self.rungame = False
        taskMgr.add(self.game, "Game",)

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
        if self.rungame:
            if self.GameOver == False:
                if self.new_trial:
                    self.init_trial()
                elif self.go:
                    self.godisp()
                else:
                    if self.joystickon == 1:
                        self.getjoystick()
                    else:
                        self.move()
                    self.animate()
                    self.dig()
                    self.drag()
                self.movecam()
                if self.savelog:
                    self.log()
                    self.savelog = False
            else:
                if self.BlackScreen == 0:
                    self.BlackScreen = 1
                    OnscreenImage(image = 'textures/black.jpg', pos = (0, 0, 0),scale=100)
                    self.starimage = OnscreenImage(image = 'textures/goldstar.png', pos = (0,0,0),scale=0.1)
                    self.starimage.setTransparency(TransparencyAttrib.MAlpha)
                    #self.EndText = OnscreenText(text = "GREAT JOB!" , pos = (-100, -0.5), scale = 0.1*self.goscale)
        return task.cont

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




    def move(self):
        dt = globalClock.getDt() #get time from last frame
        #check for keyboard input
        if self.canMove == True:
            self.isMoving=False
            if self.LRSpin == "Spin":
                if self.keyMap["spin"]:
                    pass
                    #self.camspin = True
                else:
                    self.camspin = False
                if self.keyMap["left"]:
                    self.kiki.setH(self.kiki.getH() + 300 * dt)
                    self.isTurning = True
                if self.keyMap["right"]:
                    self.kiki.setH(self.kiki.getH() - 300 * dt)
                    self.isTurning = True
                if self.keyMap["forward"]:
                    self.isMoving=True
                    self.get_move(dt)
                if self.keyMap["back"]:
                    self.isMoving=True
                    self.get_move(dt,back=1)
                if not self.keyMap["left"] and not self.keyMap["right"]:
                    self.isTurning = False
            else:
                a,b = -1,-1
                self.isTurning = True
                if self.keyMap["left"]:
                    self.kiki.setH(90)
                    self.isMoving=True
                    self.get_move(dt,side=1)
                    a = 0
                if self.keyMap["right"]:
                    self.kiki.setH(270)
                    self.isMoving=True
                    self.get_move(dt,side=-1)
                    a = 1
                if self.keyMap["forward"]:
                    self.kiki.setH(0)
                    self.isMoving=True
                    self.get_move(dt)
                    b = 0
                if self.keyMap["back"]:
                    self.kiki.setH(180)
                    self.isMoving=True
                    self.get_move(dt,back=1)
                    b = 1
                if a > -1 and b > -1:
                    self.kiki.setH([[45,315],[135,225]][b][a])
                
        if self.keyMap["dig"]:
            if self.get_dig()>-1:
                self.digging = True

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
                	# if they dig, add a point and end game! 
                    if self.trialscore == 0:
                        self.TotalScore+=1
                        self.addstar = 1
                        self.trialscore = 1
                        self.RunningTotal+=1
                       # self.GameOver = True
                else: 
                    self.RunningTotal = 0
                    #self.GameOver = True
        return dig

    def dig(self):
        if self.digging == True:
            self.plnp.setPos(self.boxloca[self.presLoc[self.dug]][0],self.boxloca[self.presLoc[self.dug]][1], 0)
            self.render.setLight(self.plnp)
            self.isMoving = False
            self.canMove=False
            if self.boxcam == True:
                    if self.dug == self.trial[0]:
                        self.star.reparentTo(self.render)
                        self.starspin()
                       # self.GameOver = True ###!#!#!#!#!#!#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    elif self.negfeedbac ==1:
                        if self.clouddisp == 1:
                            self.clouddisp =0
                            ex = 0
                            self.presents[self.dug].detachNode()
                            self.clouds.reparentTo(self.render)
                            self.clouds.setPos(self.boxloca[self.presLoc[self.dug]][0],self.boxloca[self.presLoc[self.dug]][1],2)
                        if self.dragback == False:
                            if flipbook_2(self.clouds,self.cloudsframe,loop=0):
                                self.dragback= True
                                self.clouds.detachNode()
                                self.clouds.currentframe = 0
                                self.digging = False
                                
                            else:
                                flipbook_2(self.kiki,self.losingframe,loop=1)
                    else:
                        self.digging = False
                        self.dragback= True
                       # self.GameOver = True
            else:
                a = flipbook_2(self.kiki,self.digframe,0)
                flipbook_2(self.presents[self.dug],self.presentsframe,0)
                if a:
                    self.boxcam = True

    def starspin(self):
        self.presents[self.dug].detachNode()
        self.spin +=60./self.framerate
        if self.spin < 180:
            self.star.setH(self.spin*5)
            self.star.setPos(self.boxloca[self.presLoc[self.dug]][0],self.boxloca[self.presLoc[self.dug]][1],3+self.spin/30.)
            flipbook_2(self.kiki,self.winframe,loop=1)
        else: 
            self.dragback = True
            self.star.detachNode()
            self.digging = False

    def get_move(self,dt,back = 0,side = 0):
        if self.LRSpin == "Spin":
            if back == 0:
                dir = -self.kiki.getH()/360*2*np.pi
            else:
                dir = -self.kiki.getH()/360*2*np.pi-np.pi
        else:
            dir = 0
            if back == 1:
                dir = pi
            if side == -1:
                dir = pi/2
            if side == 1:
                dir = 3*pi/2
            
        prop = [self.kiki.getPos()[0]-25*np.sin(dir)*dt,self.kiki.getPos()[1]-25*np.cos(dir)*dt]
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
        else:
            if self.atbox > -1:
                self.presents[self.atbox].setTexture(self.texgrey)
                self.atbox = -1
            a = -1

        if a > -1:
            if self.atbox == -1:
                self.visits[self.presLoc.index(a)]+=1
            self.atbox = self.presLoc.index(a)
            self.presents[self.atbox].setTexture(self.textures[self.atbox])
            
        

    def animate(self):
        if self.isMoving:
            flipbook_2(self.kiki,self.runframe)
        else: self.kiki.pose("go", 0)

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
            ### THIS IS WHAT IS ACTUALLY STOPPING THE GAME AFTER ONE TRIAL
            	self.GameOver = True
               # self.new_trial = True

#### CHANGED THIS
    def init_trial(self):
        if self.RunningTotal == 1:
        #self.revafter:
        # and self.reversal == 0:
            self.GameOver = True #reversal = 1
            #self.RunningTotal = 1
 #           #self.correctbox = np.random.choice([i for i in range(3) if i != self.correctbox])
            #self.correctbox = 1
            #self.textures[-1] = self.tex5
            # could change the self.reversal to keep going until 5 reversals, for example.
        #elif self.RunningTotal == 1:
        	#self.revafter and self.reversal == 1:
         #   self.GameOver = True
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
        np.random.shuffle(self.presLoc)
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

    def initkeys(self):
        self.keyMap = {"left": 0, "right": 0, "forward": 0,"back":0, "dig":0,"spin":0}
        self.accept("escape", sys.exit)
        self.accept("arrow_left", self.setKey, ["left", True])
        self.accept("arrow_right", self.setKey, ["right", True])
        self.accept("arrow_up", self.setKey, ["forward", True])
        self.accept("arrow_down", self.setKey, ["back", True])
        self.accept("d",self.setKey, ["dig",True])
        self.accept("y",self.setKey, ["spin",True])
        self.accept("arrow_left-up", self.setKey, ["left", False])
        self.accept("arrow_right-up", self.setKey, ["right", False])
        self.accept("arrow_up-up", self.setKey, ["forward", False])
        self.accept("arrow_down-up", self.setKey, ["back", False])
        self.accept("d-up",self.setKey, ["dig",False])
        self.accept("y-up",self.setKey, ["spin",False])

    def setKey(self, key, value):
        self.keyMap[key] = value

    def initgame(self):
        self.transition = Transitions(loader)
        self.getparameters()
        
        self.GameOver = False
        self.BlackScreen = 0

        self.p_ID = self.b.get()
        self.p_age = self.d.get()
        self.p_gender = self.f.get()

        self.file = open('logs/Reversal3D_log_practice_'+str(self.p_ID)+'.txt', 'w')
        self.file.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t" % 
            ("pID","Age","Gender","TrialNumber","Reversal","Target",
            "Choice","Correct","TotalScore","RunningTotal","Visits0","Visits1",
            "Visits2","Visits3","ResponseTime"))

        print "ID:", self.p_ID, "Gender:",self.p_gender,"Age:",self.p_age


        self.rungame = True
        self.TotalScore = 0
        self.RunningTotal = 0
        self.trial_no = -1
        self.new_trial = True
        if self.p_gender == "Male":
            self.kiki = CreateActor("Models/baseboy.x","textures/baseboy2.png",0,0,0,0)
        else:
            self.kiki = CreateActor("Models/kiki.x","textures/kiki.jpg",0,0,0,0)
        self.kiki.currentframe= 0
        self.kiki.reparentTo(self.render)
        self.isTurning = False
        self.lag = 0
        self.ScoreText = None
        
        self.framerate = 30
        globalClock.setMode(ClockObject.MLimited)
        globalClock.setFrameRate(self.framerate)
        self.frames = [int(round(i)) for i in np.arange(0,1200,60./self.framerate)]

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
        
        
        if self.boxcol == "A":
            self.tex1 = loader.loadTexture("textures/presentblue.png")
            self.tex2 = loader.loadTexture("textures/presentyellow.jpg")
            self.tex3 = loader.loadTexture("textures/presentpink.jpg")
            self.tex4 = loader.loadTexture("textures/presentgreen.png")
            self.tex5 = loader.loadTexture("textures/presentpurple.png")
        else:
            self.tex1 = loader.loadTexture("textures/presentbrown.jpg")
            self.tex2 = loader.loadTexture("textures/presentorange.png")
            self.tex3 = loader.loadTexture("textures/presentred.png")
            self.tex4 = loader.loadTexture("textures/presentgreys.png")
            self.tex5 = loader.loadTexture("textures/presentlightpurple.png")
        
        self.textures = [self.tex1,self.tex2,self.tex3,self.tex4]
        
        self.star = self.loader.loadModel("Models/star.x")
        tex = loader.loadTexture("textures/gold_texture.jpg")
        self.star.setTexture(tex)
        self.star.setScale(self.starsize)
        self.star.setPos(0, 0, 3)
        
        self.clouds = CreateActor("Models/clouds.x","textures/cotton.jpg",0,0,0,0)
        self.star.setScale(self.cloudsize)
        
        self.environ = self.loader.loadModel("Models/Room.x")
        self.environ.reparentTo(self.render)
        
        if self.wallcol == "A":
            tex = loader.loadTexture("textures/arena.png")
        elif self.wallcol == "B":
            tex = loader.loadTexture("textures/room2.png")
        self.environ.setTexture(tex)
        self.environ.setScale(2, 2, 2)
        self.environ.setPos(0, 0, 0)

        self.starLoc = []
        for i in range(4):
            for j in range(10):
                self.starLoc.append([i,j])
        
        self.floater = NodePath(PandaNode("floater"))
        self.floater.reparentTo(self.kiki)
        self.floater.setZ(2.0)
        
        self.angle = -2*np.pi*self.kiki.getH()/360
        
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

        self.correctbox =0
 #       self.correctbox = np.random.randint(0,3)
        self.reversal = 0

        self.savelog = False

        self.joystickon = 0

        try:
            if joystick.get_count() > 0:
                self.joystickon = 1
                self.joystick = joystick.Joystick(0)
                self.joystick.init()
        except:
            print "Joystick Error" 
            self.joystickon = 0


    def getparameters(self):
        with open("parameters.txt",'r') as f:
            params= [x.strip().split('\t') for x in f]
        self.viewheight = int(params[1][1])
        self.change_dist = float(params[3][1])
        self.time_out = float(params[4][1])
       
       #######!!!! CHANGED THIS !!!! ####
        self.revafter = 1 #int(params[2][1])

        self.negfeedbac = {"Smoke-puff":1,"None":0}[params[5][1]]

        sizes = {"Small":0.5, "Medium":1, "Large":2}
        self.cloudsize = sizes[params[7][1]]
        self.starsize = sizes[params[6][1]]
        ####### !!! CHANGED BOX COLOR SO IT IS DIFFERENT THAN REAL THING
        
        self.boxcol = "B" #params[8][1]
        self.wallcol = "B" #params[9][1]
        self.camRot = params[10][1]
        self.LRSpin = params[11][1]



    def movecam(self):
        if self.camspin:
            self.camspinning += 1./10
            self.camera.setPos(self.kiki.getX()+(10*np.sin(self.angle+self.camspinning)), self.kiki.getY() + 10*np.cos(self.angle+self.camspinning), self.viewheight)
            self.camera.lookAt(self.floater)
        elif self.camRot == "On":
            if self.isTurning:
                self.lag=0.9
            elif self.lag > 0:
                self.lag-=(0.01)*(60./self.framerate)
            else: self.lag = 0

            newangle = -2*np.pi*self.kiki.getH()/360
            self.angle = newangle+(self.angle-newangle)*self.lag
            self.camera.setPos(self.kiki.getX()+(10*np.sin(self.angle)), self.kiki.getY() + 10*np.cos(self.angle),self.viewheight)
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
                self.camera.setPos(self.kiki.getX()+(10*np.sin(self.angle)), self.kiki.getY() + 10*np.cos(self.angle), self.viewheight)
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
        else:
            self.camera.setPos(self.kiki.getX(), self.kiki.getY()+0.1,self.viewheight)
            self.camera.lookAt(self.floater)

    def seconds(self,secs):
        return secs*self.framerate

def CreateActor(fileName,texture,x,y,z,poseframe,):
    actor = Actor(fileName,{"go":fileName})
    tex = loader.loadTexture(texture)
    actor.setTexture(tex)
    actor.setPos(x,y,z)
    actor.pose("go", poseframe)
    actor.currentframe=poseframe
    return actor

def get_frames(start,end,framerate):
    return [int(round(i)) for i in np.arange(start,end,60./framerate)]

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

def button(name,y,x=0,extras=[]):
    b = DirectButton(text = name, scale=.1, command=check_vars,extraArgs = extras)
    return b

if __name__ == "__main__":
    main()
