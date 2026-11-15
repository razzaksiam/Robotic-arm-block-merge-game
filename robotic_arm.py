from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import math
import random
import sys

W, H = 900, 600            # logical world size
ww, wh = W, H              # real window size (updated in reshape)

FLOOR = 120                # floor level (button panel is below it)
BX, BY = 450, 170          # shoulder joint position
L1, L2, L3 = 170, 140, 40  # upper arm, forearm, wrist lengths
FING = 36                  # finger length
TIP = L3 + 20              # wrist joint -> grip point distance
BOX = 40                   # block size
MAX_BOXES = 30
OPEN_GAP = 34              # gripper half-gap when open

COLORS = [(0.92, 0.26, 0.28), (0.25, 0.62, 0.92), (0.98, 0.80, 0.18),
          (0.32, 0.78, 0.40), (0.72, 0.40, 0.88), (0.97, 0.58, 0.20)]
SHOW_SCORE = True          # set False to hide the score

a1, a2, a3 = 90.0, -60.0, -30.0      # current joint angles (degrees)
t1, t2, t3 = a1, a2, a3              # target joint angles
g, tg = OPEN_GAP, OPEN_GAP           # gripper half-gap, current / target
target = None                        # last mouse target (red cross)
holding = None                       # index of held block
boxes = []
buttons = []
pressed = None                       # button currently held down
hover = None                         # button under the mouse
ik_mode = False                      # True while moving towards a mouse target
drag_target = False                  # True while dragging the arm target
stall = 0
score = 0
particles = []                       # merge effects
popups = []                          # floating +score texts

_rng = random.Random(7)
STARS = [(_rng.randint(0, W), _rng.randint(FLOOR + 60, H), _rng.random()) for _ in range(70)]


def reset():
    global a1, a2, a3, t1, t2, t3, g, tg, target, holding, boxes, ik_mode, stall
    global score, particles, popups
    a1, a2, a3 = 90.0, -60.0, -30.0
    t1, t2, t3 = a1, a2, a3
    g, tg = OPEN_GAP, OPEN_GAP
    target = None
    holding = None
    ik_mode = False
    stall = 0
    score = 0
    particles = []
    popups = []
    y0 = FLOOR + BOX / 2
    slots = []
    for xs in ([105, 165, 225, 285], [615, 675, 735, 795]):
        slots += [(xs[0], y0), (xs[1], y0), (xs[2], y0), (xs[3], y0),
                  (xs[0], y0 + BOX), (xs[1], y0 + BOX), (xs[2], y0 + BOX),
                  (xs[1], y0 + 2 * BOX)]
    while True:                       # random colours, but nothing already merged
        boxes = [[x, y, random.choice(COLORS)] for x, y in slots]
        if not find_groups(set()):
            break


# ---------------- helpers ----------------
def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def shade(c, f):
    return tuple(clamp(v * f, 0, 1) for v in c)


def rect(x1, y1, x2, y2):
    glBegin(GL_QUADS)
    glVertex2f(x1, y1)
    glVertex2f(x2, y1)
    glVertex2f(x2, y2)
    glVertex2f(x1, y2)
    glEnd()


def rect_outline(x1, y1, x2, y2, w=1):
    glLineWidth(w)
    glBegin(GL_LINE_LOOP)
    glVertex2f(x1, y1)
    glVertex2f(x2, y1)
    glVertex2f(x2, y2)
    glVertex2f(x1, y2)
    glEnd()


def circle(cx, cy, r):
    glBegin(GL_TRIANGLE_FAN)
    glVertex2f(cx, cy)
    for i in range(41):
        ang = 2 * math.pi * i / 40
        glVertex2f(cx + r * math.cos(ang), cy + r * math.sin(ang))
    glEnd()


def text(x, y, s):
    glRasterPos2f(x, y)
    for ch in s:
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, ord(ch))


def text_width(s):
    try:
        return glutBitmapLength(GLUT_BITMAP_HELVETICA_12, s.encode())
    except Exception:
        return 7 * len(s)


# ---------------- kinematics ----------------
def fk(A1=None, A2=None, A3=None):
    """Forward kinematics: returns elbow, wrist, tip positions."""
    A1 = a1 if A1 is None else A1
    A2 = a2 if A2 is None else A2
    A3 = a3 if A3 is None else A3
    r1 = math.radians(A1)
    r2 = math.radians(A1 + A2)
    r3 = math.radians(A1 + A2 + A3)
    ex, ey = BX + L1 * math.cos(r1), BY + L1 * math.sin(r1)
    wx, wy = ex + L2 * math.cos(r2), ey + L2 * math.sin(r2)
    tx, ty = wx + TIP * math.cos(r3), wy + TIP * math.sin(r3)
    return (ex, ey), (wx, wy), (tx, ty)


def lowest_y(A1, A2, A3, G):
    """Lowest point of the arm (links, joints, fingers, held block)."""
    (ex, ey), (wx, wy), (tx, ty) = fk(A1, A2, A3)
    r3 = math.radians(A1 + A2 + A3)
    dx, dy = math.cos(r3), math.sin(r3)
    nx, ny = -dy, dx
    bx, by = wx + dx * L3, wy + dy * L3
    ex2, ey2 = bx + dx * FING, by + dy * FING
    ys = [ey - 13, wy - 10]
    for s in (-1, 1):
        for off in (G, G + 6):
            ys.append(by + s * ny * off)
            ys.append(ey2 + s * ny * off)
    if holding is not None:
        ys.append(ty - BOX / 2)
    return min(ys)


def ok(A1, A2, A3, G):
    return lowest_y(A1, A2, A3, G) >= FLOOR + 1


def inverse_kinematics(x, y):
    """Move the gripper to a mouse position using two-link IK."""
    global t1, t2, t3
    y = max(y, FLOOR + BOX / 2 + 2)        # never go below the floor line
    wx, wy = x, y + TIP
    dx, dy = wx - BX, wy - BY
    d = math.hypot(dx, dy)
    d = clamp(d, abs(L1 - L2) + 1, L1 + L2 - 1)
    base = math.atan2(dy, dx)
    c = (d * d - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    elbow = -math.acos(clamp(c, -1, 1))
    shoulder = base - math.atan2(L2 * math.sin(elbow), L1 + L2 * math.cos(elbow))
    t1 = clamp(math.degrees(shoulder), 0, 180)
    t2 = clamp(math.degrees(elbow), -150, 150)
    t3 = clamp(-90 - (t1 + t2), -120, 120)
    return y


# ---------------- actions ----------------
def joint(j, d):
    global t1, t2, t3, ik_mode
    ik_mode = False
    if j == 1:
        t1 = clamp(t1 + d, 0, 180)
    elif j == 2:
        t2 = clamp(t2 + d, -150, 150)
    else:
        t3 = clamp(t3 + d, -120, 120)


def grip_open():
    global tg
    tg = OPEN_GAP


def grip_close():
    global tg
    tg = BOX / 2 + 1 if holding is not None else 4


def toggle_grab():
    global holding, tg
    _, _, tip = fk()
    if holding is not None:            # release
        holding = None
        tg = OPEN_GAP
        return
    best, bd = None, 50
    for i, b in enumerate(boxes):      # nearest block to the grip point
        d = math.hypot(b[0] - tip[0], b[1] - tip[1])
        if d < bd:
            best, bd = i, d
    if best is not None:
        holding = best
        tg = BOX / 2 + 1


def spawn_block(avoid=(), k=0):
    """Drop a new block (a colour different from the merged ones) from above the screen."""
    if len(boxes) >= MAX_BOXES:
        return
    choices = [c for c in COLORS if c not in avoid] or COLORS
    x = random.choice([random.randint(100, 330), random.randint(570, 800)])
    boxes.append([x, H + 30 + k * BOX * 1.6, random.choice(choices)])


def new_block():
    spawn_block()


# ---------------- merging ----------------
def touching(b, o):
    if b[2] != o[2]:
        return False
    dx, dy = abs(b[0] - o[0]), abs(b[1] - o[1])
    return (dy <= 3 and dx <= BOX + 6) or (dx < BOX and dy <= BOX + 4)


def find_groups(skip):
    """Groups (size >= 2) of same-coloured blocks that touch each other."""
    seen, groups = set(), []
    for i in range(len(boxes)):
        if i in seen or i in skip:
            continue
        grp, queue = [], [i]
        seen.add(i)
        while queue:
            c = queue.pop()
            grp.append(c)
            for j in range(len(boxes)):
                if j not in seen and j not in skip and touching(boxes[c], boxes[j]):
                    seen.add(j)
                    queue.append(j)
        if len(grp) >= 2:
            groups.append(grp)
    return groups


def burst(x, y, c):
    for _ in range(16):
        ang = random.uniform(0, 2 * math.pi)
        sp = random.uniform(1.0, 4.5)
        particles.append([x + random.uniform(-12, 12), y + random.uniform(-12, 12),
                          math.cos(ang) * sp, math.sin(ang) * sp + 1.5,
                          random.randint(30, 55), c])


def process_merges(falling):
    global boxes, holding, score
    skip = set(falling)
    if holding is not None:
        skip.add(holding)
    groups = find_groups(skip)
    if not groups:
        return
    remove, avoid, spawn = set(), set(), 0
    for grp in groups:
        n = len(grp)
        col = boxes[grp[0]][2]
        cx = sum(boxes[i][0] for i in grp) / n
        cy = sum(boxes[i][1] for i in grp) / n
        gain = 10 * n * (n - 1)
        score += gain
        popups.append([cx, cy + 20, 60, "+%d" % gain, col])
        for i in grp:
            burst(boxes[i][0], boxes[i][1], col)
        remove |= set(grp)
        avoid.add(col)
        spawn += n
    if holding is not None:
        holding -= sum(1 for i in remove if i < holding)
    boxes = [b for i, b in enumerate(boxes) if i not in remove]
    for k in range(spawn):
        spawn_block(avoid, k)


# ---------------- buttons ----------------
def build_buttons():
    global buttons
    buttons = []
    row1 = [("Shoulder +", lambda: joint(1, 2)), ("Shoulder -", lambda: joint(1, -2)),
            ("Elbow +", lambda: joint(2, 2)), ("Elbow -", lambda: joint(2, -2)),
            ("Wrist +", lambda: joint(3, 2)), ("Wrist -", lambda: joint(3, -2))]
    x = 70
    for label, fn in row1:
        buttons.append({'label': label, 'x': x, 'y': 74, 'w': 120, 'h': 32,
                        'hold': True, 'fn': fn})
        x += 128
    row2 = [("Open", grip_open), ("Close", grip_close), ("Grab / Release", toggle_grab),
            ("New Block", new_block), ("Reset", reset)]
    x = 84
    for label, fn in row2:
        buttons.append({'label': label, 'x': x, 'y': 28, 'w': 140, 'h': 32,
                        'hold': False, 'fn': fn})
        x += 148


def hit_button(mx, my):
    for b in buttons:
        if b['x'] <= mx <= b['x'] + b['w'] and b['y'] <= my <= b['y'] + b['h']:
            return b
    return None


def draw_buttons():
    for b in buttons:
        x, y, w, h = b['x'], b['y'], b['w'], b['h']
        if b is pressed:
            top, bot = (0.20, 0.50, 0.85), (0.14, 0.38, 0.70)
        elif b is hover:
            top, bot = (0.45, 0.50, 0.62), (0.33, 0.37, 0.48)
        else:
            top, bot = (0.36, 0.40, 0.50), (0.24, 0.27, 0.35)
        glColor4f(0, 0, 0, 0.35)                       # shadow
        rect(x + 2, y - 3, x + w + 2, y + h - 3)
        glBegin(GL_QUADS)                              # gradient body
        glColor3f(*top)
        glVertex2f(x, y + h)
        glVertex2f(x + w, y + h)
        glColor3f(*bot)
        glVertex2f(x + w, y)
        glVertex2f(x, y)
        glEnd()
        glColor4f(1, 1, 1, 0.12)                       # gloss
        rect(x, y + h - 3, x + w, y + h)
        glColor4f(1, 1, 1, 0.35)
        rect_outline(x, y, x + w, y + h)
        glColor3f(1, 1, 1)
        text(x + (w - text_width(b['label'])) / 2, y + 11, b['label'])


# ---------------- drawing ----------------
def draw_link(length, width, top, bottom):
    glBegin(GL_QUADS)
    glColor3f(*top)
    glVertex2f(0, width)
    glVertex2f(length, width)
    glColor3f(*bottom)
    glVertex2f(length, -width)
    glVertex2f(0, -width)
    glEnd()
    glColor3f(*shade(top, 0.95))
    circle(0, 0, width)
    circle(length, 0, width)
    glColor4f(1, 1, 1, 0.28)                           # highlight stripe
    rect(0, width * 0.35, length, width * 0.65)


def hub(r):
    glColor3f(0.20, 0.21, 0.25)
    circle(0, 0, r + 3)
    glColor3f(0.85, 0.87, 0.92)
    circle(0, 0, r)
    glColor3f(0.55, 0.58, 0.65)
    circle(0, 0, r * 0.55)
    glColor3f(0.2, 0.2, 0.25)
    circle(0, 0, r * 0.2)


def draw_arm():
    # pedestal
    glColor4f(0, 0, 0, 0.3)
    rect(BX - 82, FLOOR - 4, BX + 90, FLOOR + 6)
    glColor3f(0.22, 0.24, 0.30)
    rect(BX - 85, FLOOR, BX + 85, FLOOR + 14)
    glBegin(GL_QUADS)
    glColor3f(0.40, 0.43, 0.52)
    glVertex2f(BX - 55, BY - 20)
    glVertex2f(BX + 55, BY - 20)
    glColor3f(0.28, 0.30, 0.38)
    glVertex2f(BX + 70, FLOOR + 14)
    glVertex2f(BX - 70, FLOOR + 14)
    glEnd()
    glColor3f(0.55, 0.58, 0.68)
    rect(BX - 45, BY - 20, BX + 45, BY)
    glColor3f(0.8, 0.82, 0.9)
    for bx in (-70, 70):                               # bolts on base plate
        circle(BX + bx, FLOOR + 7, 3)

    glPushMatrix()
    glTranslatef(BX, BY, 0)
    glRotatef(a1, 0, 0, 1)
    draw_link(L1, 13, (0.30, 0.62, 0.98), (0.10, 0.34, 0.72))
    hub(17)

    glTranslatef(L1, 0, 0)
    glRotatef(a2, 0, 0, 1)
    draw_link(L2, 10, (0.38, 0.85, 0.52), (0.12, 0.55, 0.30))
    hub(14)

    glTranslatef(L2, 0, 0)
    glRotatef(a3, 0, 0, 1)
    draw_link(L3, 7, (1.0, 0.70, 0.25), (0.80, 0.42, 0.05))
    hub(10)

    glTranslatef(L3, 0, 0)                             # gripper
    glColor3f(0.62, 0.65, 0.72)
    rect(0, -g - 7, 9, g + 7)
    glColor3f(0.85, 0.87, 0.92)
    rect(0, -g - 7, 9, -g + 2)
    rect(0, g - 2, 9, g + 7)
    for s in (1, -1):
        glColor3f(0.78, 0.80, 0.86)
        rect(0, min(s * g, s * (g + 7)), FING, max(s * g, s * (g + 7)))
        glColor3f(0.30, 0.32, 0.38)                    # rubber pad
        rect(FING - 14, min(s * (g - 3), s * g), FING, max(s * (g - 3), s * g))
    glPopMatrix()


def draw_scene():
    glBegin(GL_QUADS)                                  # sky gradient
    glColor3f(0.05, 0.07, 0.16)
    glVertex2f(0, H)
    glVertex2f(W, H)
    glColor3f(0.32, 0.42, 0.65)
    glVertex2f(W, FLOOR)
    glVertex2f(0, FLOOR)
    glEnd()
    glPointSize(2)                                     # stars
    glBegin(GL_POINTS)
    for sx, sy, br in STARS:
        glColor4f(1, 1, 1, 0.3 + 0.6 * br)
        glVertex2f(sx, sy)
    glEnd()
    glColor4f(0.10, 0.14, 0.28, 0.85)                  # far mountains
    glBegin(GL_TRIANGLES)
    for cx, hw, hh in [(100, 200, 130), (330, 170, 90), (560, 220, 150), (800, 220, 110)]:
        glVertex2f(cx - hw, FLOOR)
        glVertex2f(cx + hw, FLOOR)
        glVertex2f(cx, FLOOR + hh)
    glEnd()
    glColor3f(0.17, 0.18, 0.23)                        # floor
    rect(0, 0, W, FLOOR)
    glColor3f(0.24, 0.26, 0.32)
    rect(0, FLOOR - 14, W, FLOOR)
    glColor3f(0.95, 0.80, 0.15)                        # hazard stripe
    rect(0, FLOOR - 4, W, FLOOR)
    glColor3f(0.1, 0.1, 0.1)
    x = 0
    while x < W:
        glBegin(GL_QUADS)
        glVertex2f(x, FLOOR - 4)
        glVertex2f(x + 8, FLOOR - 4)
        glVertex2f(x + 14, FLOOR)
        glVertex2f(x + 6, FLOOR)
        glEnd()
        x += 24
    glColor4f(1, 1, 1, 0.08)
    rect(0, FLOOR - 15, W, FLOOR - 14)


def draw_boxes():
    h = BOX / 2
    for b in boxes:
        x, y, c = b
        glColor4f(0, 0, 0, 0.28)                       # drop shadow
        rect(x - h + 4, y - h - 3, x + h + 4, y + h - 3)
        glColor3f(*c)
        rect(x - h, y - h, x + h, y + h)
        glColor3f(*shade(c, 1.3))                      # top bevel
        rect(x - h, y + h - 6, x + h, y + h)
        glColor3f(*shade(c, 0.65))                     # bottom bevel
        rect(x - h, y - h, x + h, y - h + 6)
        glColor3f(*shade(c, 0.8))                      # inner square
        rect_outline(x - h + 9, y - h + 9, x + h - 9, y + h - 9, 2)
        glColor3f(0.05, 0.05, 0.08)
        rect_outline(x - h, y - h, x + h, y + h, 1.5)


def draw_effects():
    for x, y, vx, vy, life, c in particles:
        glColor4f(c[0], c[1], c[2], min(1.0, life / 30.0))
        rect(x - 3, y - 3, x + 3, y + 3)
    for x, y, life, label, c in popups:
        glColor4f(1, 1, 1, min(1.0, life / 30.0))
        text(x - text_width(label) / 2, y, label)
    if SHOW_SCORE:
        glColor3f(1, 1, 1)
        label = "Score: %d" % score
        text(W - text_width(label) - 16, H - 24, label)


def draw_target():
    if target:
        x, y = target
        glColor4f(1, 0.3, 0.3, 0.9)
        glLineWidth(2)
        glBegin(GL_LINES)
        glVertex2f(x - 9, y)
        glVertex2f(x + 9, y)
        glVertex2f(x, y - 9)
        glVertex2f(x, y + 9)
        glEnd()


# ---------------- physics / animation ----------------
def update_physics():
    falling = set()
    order = sorted(range(len(boxes)), key=lambda i: boxes[i][1])
    for i in order:
        if i == holding:
            continue
        b = boxes[i]
        support = FLOOR + BOX / 2
        for j, o in enumerate(boxes):
            if j != i and abs(o[0] - b[0]) < BOX and o[1] < b[1]:
                support = max(support, o[1] + BOX)
        if b[1] > support:
            b[1] = max(support, b[1] - 6)
            falling.add(i)
    return falling


def step(cur, goal, s):
    if abs(goal - cur) <= s:
        return goal
    return cur + s if goal > cur else cur - s


def update(value):
    global a1, a2, a3, g, t1, t2, t3, tg, stall
    if pressed is not None and pressed['hold']:
        pressed['fn']()
    moved = False

    # Move all three joints together for a smoother arm path.
    changes = [t1 - a1, t2 - a2, t3 - a3]
    largest_change = max(abs(change) for change in changes)
    if largest_change > 0.01:
        fraction = min(1.0, 2.0 / largest_change)
        next_angles = [a1 + changes[0] * fraction,
                       a2 + changes[1] * fraction,
                       a3 + changes[2] * fraction]

        # Check the full move against the floor before applying it.
        if ok(next_angles[0], next_angles[1], next_angles[2], g):
            a1, a2, a3 = next_angles
            moved = True
        else:
            # Try a smaller step if the normal step is blocked.
            fraction *= 0.5
            next_angles = [a1 + changes[0] * fraction,
                           a2 + changes[1] * fraction,
                           a3 + changes[2] * fraction]
            if ok(next_angles[0], next_angles[1], next_angles[2], g):
                a1, a2, a3 = next_angles
                moved = True
            elif not ik_mode:
                t1, t2, t3 = a1, a2, a3
    n = step(g, tg, 1.5)
    if n != g:
        if ok(a1, a2, a3, n):
            g, moved = n, True
        else:
            tg = g

    pending = (t1 != a1 or t2 != a2 or t3 != a3 or tg != g)
    stall = stall + 1 if (pending and not moved) else 0
    if stall > 20:                                     # blocked by the floor: give up
        t1, t2, t3, tg = a1, a2, a3, g
        stall = 0

    if holding is not None:
        _, _, tip = fk()
        boxes[holding][0], boxes[holding][1] = tip
    falling = update_physics()
    process_merges(falling)
    for p in particles:
        p[0] += p[2]
        p[1] += p[3]
        p[3] -= 0.15
        p[4] -= 1
    particles[:] = [p for p in particles if p[4] > 0]
    for q in popups:
        q[1] += 0.7
        q[2] -= 1
    popups[:] = [q for q in popups if q[2] > 0]
    glutPostRedisplay()
    glutTimerFunc(16, update, 0)


# ---------------- input ----------------
def keyboard(key, x, y):
    try:
        k = key.decode().lower()
    except Exception:
        return
    if k in ('\r', '\n'):              # Enter = grab / release
        toggle_grab()
    elif k == 'a':
        joint(1, 4)
    elif k == 'd':
        joint(1, -4)
    elif k == 'w':
        joint(2, 4)
    elif k == 's':
        joint(2, -4)
    elif k == 'q':
        joint(3, 4)
    elif k == 'e':
        joint(3, -4)
    elif k == 'o':
        grip_open()
    elif k == 'c':
        grip_close()
    elif k == ' ':
        toggle_grab()
    elif k == 'n':
        new_block()
    elif k == 'r':
        reset()
    elif k == '\x1b':
        sys.exit(0)


def special_key(key, x, y):
    if key == GLUT_KEY_LEFT:
        joint(1, 4)
    elif key == GLUT_KEY_RIGHT:
        joint(1, -4)
    elif key == GLUT_KEY_UP:
        joint(2, 4)
    elif key == GLUT_KEY_DOWN:
        joint(2, -4)
    elif key == GLUT_KEY_PAGE_UP:
        joint(3, 4)
    elif key == GLUT_KEY_PAGE_DOWN:
        joint(3, -4)


def mouse(button, state, x, y):
    global target, pressed, ik_mode, drag_target
    mx = x * W / ww
    my = (wh - y) * H / wh
    if state == GLUT_DOWN:
        b = hit_button(mx, my)
        if b is not None and button == GLUT_LEFT_BUTTON:
            pressed = b
            b['fn']()
        elif button == GLUT_LEFT_BUTTON and my > FLOOR:
            # Set a target and start dragging it with the mouse.
            drag_target = True
            ik_mode = True
            ty = inverse_kinematics(mx, my)
            target = (mx, ty)
        return
    was_button = pressed is not None
    pressed = None
    drag_target = False
    if was_button:
        return
    if button == GLUT_RIGHT_BUTTON:
        toggle_grab()


def motion(x, y):
    global hover, target, ik_mode
    mx = x * W / ww
    my = (wh - y) * H / wh
    hover = hit_button(mx, my)
    if drag_target:
        # Keep calculating arm angles while the mouse is moving.
        ty = inverse_kinematics(mx, my)
        target = (mx, ty)
        ik_mode = True


def reshape(width, height):
    global ww, wh
    ww, wh = width, max(height, 1)
    glViewport(0, 0, ww, wh)


# ---------------- display ----------------
def showScreen():
    glClear(GL_COLOR_BUFFER_BIT)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluOrtho2D(0, W, 0, H)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    draw_scene()
    draw_target()
    draw_boxes()
    draw_arm()
    draw_effects()
    draw_buttons()
    glutSwapBuffers()


reset()
build_buttons()
glutInit()
glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE)
glutInitWindowSize(W, H)
glutInitWindowPosition(100, 50)
glutCreateWindow(b"Robotic Arm Simulator")
glEnable(GL_BLEND)
glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
glEnable(GL_LINE_SMOOTH)
glutDisplayFunc(showScreen)
glutReshapeFunc(reshape)
glutKeyboardFunc(keyboard)
glutSpecialFunc(special_key)
glutMouseFunc(mouse)
glutPassiveMotionFunc(motion)
glutMotionFunc(motion)
glutTimerFunc(16, update, 0)
glutMainLoop()
