import pygame
import sys,time
from pygame.locals import *
import random

from pygame.sprite import AbstractGroup

# pygame初始化
pygame.init()

# 设定坐标维度
vec = pygame.math.Vector2  # 2 for two dimensional
 

class Player(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__() 
        # 创建serface对象，尺寸为(30,30)
        # 创建移动的方块
        self.surf = pygame.Surface((30, 30))
        self.surf.fill((128,255,40))
        self.rect = self.surf.get_rect(center = (10, 420))

        # 设定坐标，初速度，加速度
        self.pos = vec((10, 380))
        self.vel = vec(0,0)
        self.acc = vec(0,0)

        self.jumping=False
    def move(self):
        # y轴添加加速度模拟重力
        self.acc = vec(0,0.5)
    
        pressed_keys = pygame.key.get_pressed()
                
        if pressed_keys[K_LEFT]:
            self.acc.x = -ACC
        if pressed_keys[K_RIGHT]:
            self.acc.x = ACC       

        # a=v*(-0.12)，加速度最后体现在速度上存在一定的衰减
        self.acc.x += self.vel.x * FRIC

        # v=at
        self.vel += self.acc

        # x=vt+1/2*a*t^2
        self.pos += self.vel + 0.5 * self.acc

        # 越界后从另外一侧出现
        if self.pos.x > WIDTH:
            self.pos.x = 0
        if self.pos.x < 0:
            self.pos.x = WIDTH
            
        # 设定矩形框的中点是当前坐标
        self.rect.midbottom = self.pos

    def update(self):
        # 设定碰撞判定
        hits=pygame.sprite.spritecollide(P1,platforms,False)
        if P1.vel.y>0:
            if hits:
                self.pos.y=hits[0].rect.top+1
                self.vel.y=0

       
    def jump(self):
        # 设定为仅当物体落地时才可以跳起来
        hits=pygame.sprite.spritecollide(P1,platforms,False)
        if hits:
            self.vel.y=-15
            self.jumping=True

    def cancel_jump(self):
        if self.jumping:
            if self.vel.y < -3:
                self.vel.y = -3


 
class platform(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        # 创建地面
        self.surf = pygame.Surface((WIDTH, 20))
        # 设置填充颜色RGB=255,0,0
        self.surf.fill((255,0,0))
        # 设置为矩形
        self.rect = self.surf.get_rect(center = (WIDTH/2, HEIGHT - 10))
    

class float_Platform(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.surf = pygame.Surface((random.randint(50,100), 12))
        self.surf.fill((0,255,0))
        self.rect = self.surf.get_rect(center = (random.randint(0,WIDTH-10),
                                                 random.randint(0, HEIGHT-30)))

def plat_gen():
    while len(platforms) < 7 :
        width = random.randrange(50,100)
        
        p  = float_Platform()             
        p.rect.center = (random.randrange(0, WIDTH - width),random.randrange(-50, 0))
       

        platforms.add(p)
        all_sprites.add(p)


        
# config the shape of Windows
HEIGHT = 450
WIDTH = 400
ACC = 0.5
FRIC = -0.12
FPS = 60

# 图片设定帧数记录
FramePerSec = pygame.time.Clock()

# 实例化地板、玩家对象
PT1 = platform()
P1 = Player()

# PT group创建
platforms = pygame.sprite.Group()
platforms.add(PT1)

# 添加精灵到精灵组内
all_sprites = pygame.sprite.Group()
all_sprites.add(PT1)
all_sprites.add(P1)

# 创建漂浮的砖块
for x in range(random.randint(5, 6)):
    pl = float_Platform()
    platforms.add(pl)
    all_sprites.add(pl)


# exec the configs
displaysurface = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Ship_env")



while True:	
    P1.move()
    P1.update()
    plat_gen()
    # 检测是否关闭游戏
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
         # 按下空格键的时候跳起
        if event.type == pygame.KEYDOWN:    
            if event.key == pygame.K_SPACE:
                P1.jump()

        if event.type == pygame.KEYUP:    
            if event.key == pygame.K_SPACE:
                P1.cancel_jump()

    if P1.rect.top > HEIGHT:
        for entity in all_sprites:
            entity.kill()
            time.sleep(1)
            displaysurface.fill((255,0,0))
            pygame.display.update()
            time.sleep(1)
            pygame.quit()
            sys.exit()
    
    # 设定背景颜色为黑色
    displaysurface.fill((0,0,0))

    # 将各精灵导入display surface中
    for entity in all_sprites:
        displaysurface.blit(entity.surf, entity.rect)

    if P1.rect.top <= HEIGHT / 3:
        P1.pos.y += abs(P1.vel.y)
        for plat in platforms:
            plat.rect.y += abs(P1.vel.y)
            if plat.rect.top >= HEIGHT:
                plat.kill()


    # 时刻更新
    pygame.display.update()

    FramePerSec.tick(FPS)