import direct.directbase.DirectStart
from direct.gui.OnscreenText import OnscreenText 
from direct.gui.DirectGui import *
from panda3d.core import *
import datetime
 
#callback function to set  text 
def setText(textEntered):
    pass
 
#clear the text
def clearText():
    b.enterText('')

def text_entry(name,y,ax=-0.7,bx = 0, init="",options = [],initem = 0):
    a = OnscreenText(text = name, pos = (ax,y),
            scale = 0.07,fg=(1,0.5,0.5,1),align=TextNode.ALeft,mayChange=1)
    if len(options)<1:
        b = DirectEntry(text = "" ,scale=.05,pos = LVecBase3f(bx,0,y),command=setText,
                initialText=init, numLines = 1,focus=0)
    else:
        b = DirectOptionMenu(text=options[0], scale=0.05,pos = LVecBase3f(bx,0,y),items=options,initialitem=initem,
                highlightColor=(1,0.5,0.5,1))

    return [a,b]

def checks(name,options,x,y):
    a = OnscreenText(text = name, pos = (x,y),
            scale = 0.07,fg=(1,0.5,0.5,1),align=TextNode.ALeft,mayChange=1)
    
    buttons =[DirectCheckButton(text = options[i] ,scale=0.05,pos = LVecBase3f(x+1.2+0.25*i,0,y)) for i in range(len(options))] 
    return buttons

def button(name,y,x=0,extras=[],scale = 0.07):
    b = DirectButton(text = name, scale=scale, command=check_vars,extraArgs = extras,pos = LVecBase3f(x,0,y))
    return b

def check_vars(height,corr,dist,tout,neg,smoks,stars,boxcol,wallcol,camspin,lrspin,outof):
    cont = 1
    try:
        hei = int(height[1].get())
        print hei
    except:
        cont = -1

    try:
        corre = int(corr[1].get())
        print corre
    except:
        cont = -1

    try:
        distance = float(dist[1].get())
        print distance
    except:
        cont = -1

    try:
        timeo = int(tout[1].get())
        print timeo
    except:
        cont = -1

    print smoks[1].get()

    if cont == 1:
        save_file([height,corr,dist,tout,neg,smoks,stars,boxcol,wallcol,camspin,lrspin,outof])

def save_file(input):
    today = datetime.datetime.now()
    fil1 = open("parameters.txt", 'w')
    fil2 = open("paramhistory/parameters_"+str(today.year)+'_'+str(today.month)+'_'+ str(today.day)+"__" + str(today.hour)+"_"+str(today.minute)+"_"+str(today.second) +".txt", 'w')
    for fil in [fil1,fil2]:
        fil.write("Parameters for Reversal 3D. Saved on "+str(today.day)+'/'+str(today.month)+'/'+str(today.year) + " at " + str(today.hour)+":"+str(today.minute)+":"+str(today.second))
        for i in input:
            fil.write("\n")
            fil.write("%s\t" % i[0].getText())
            fil.write("%s\t" % i[1].get())


def main():
    #add button
    with open("parameters.txt",'r') as f:
            params= [x.strip().split('\t') for x in f]


    b = text_entry("View Height",0.6,-1.3,-0.3,init = params[1][1])
    d = text_entry("#Correct before Reversal...",0.5,-1.3,-0.3,init = params[2][1])
    y = text_entry("...Out of",0.4,-1.3,-0.3,init = params[12][1])
    f = text_entry("Approach distance",0.3,-1.3,-0.3,init = params[3][1])
    h = text_entry("Time-out time",0.2,-1.3,-0.3,init = params[4][1])
    j = text_entry("Negative feedback",0.1,-1.3,-0.3,options = ["None","Smoke-puff"],initem = ["None","Smoke-puff"].index(params[5][1]))
    l = text_entry("Smoke-puff size",0.0,-1.3,-0.3,options = ["Small", "Medium", "Large"],initem = ["Small", "Medium", "Large"].index(params[6][1]))
    n = text_entry("Star size",-0.1,-1.3,-0.3,options = ["Small", "Medium", "Large"],initem = ["Small", "Medium", "Large"].index(params[7][1]))
    p = text_entry("Box Colours",-0.2,-1.3,-0.3,options = ["A", "B"],initem = ["A", "B"].index(params[8][1]))
    s = text_entry("Wall Colours",-0.3,-1.3,-0.3,options = ["A", "B"],initem = ["A", "B"].index(params[9][1]))
    u = text_entry("Camera Rotation",-0.4,-1.3,-0.3,options = ["On", "Off"],initem = ["On", "Off"].index(params[10][1]))
    w = text_entry("L/R buttons",-0.5,-1.3,-0.3,options = ["Spin", "Move"],initem = ["Spin", "Move"].index(params[11][1]))
        
    butt = button("Save",-0.7,extras=[b,d,f,h,j,l,n,p,s,u,w,y])
    base.run()

if __name__ == "__main__":
    main()