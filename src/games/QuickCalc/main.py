# main.py
import sys, argparse, time
import pygame
import pygame_gui
import math

from mechanics import load_mechanics
from engine import QuickCalcEngine, SYMBOL  # SYMBOL maps ops to + − × ÷

# ----------------- CONFIG YOU CAN TWEAK -----------------
MECH_CSV  = "data/QuickCalc/modelling/item_mechanics.csv"
NUM_LIVES = 5
BALLOON_SIZE = 2.0   
# --------------------------------------------------------

# --------- Balloon drawing ----------
def draw_balloon(surface, engine, b, x_left, channel_w, top_y, channel_h, font):
    frac = engine.balloon_y_fraction(b)
    y = int((1.0 - frac) * channel_h) + top_y
    x = x_left + channel_w // 2

    # balloon size scales with BALLOON_SIZE
    radius = int(max(26, channel_w * 0.14) * BALLOON_SIZE)

    rim_color   = (218, 120, 50)
    fill_color  = (253, 245, 237)
    text_color  = (25, 25, 25)

    pygame.draw.circle(surface, fill_color, (x, y), radius)
    pygame.draw.circle(surface, rim_color,  (x, y), radius, 6)

    op_sym = SYMBOL.get(b.operation, b.operation)
    text_str = f"{b.left}{op_sym} {b.right}" if b.operation == "percentage" else f"{b.left} {op_sym} {b.right}"

    label_font = pygame.font.SysFont(None, int(radius*0.7), bold=False)
    text = label_font.render(text_str, True, text_color)
    surface.blit(text, text.get_rect(center=(x, y)))

def draw_round_rect(surf, rect, color, radius=12, border=0, border_color=(0,0,0)):
    x, y, w, h = rect
    r = min(radius, w//2, h//2)
    s = pygame.Surface((w, h), pygame.SRCALPHA)
    pygame.draw.rect(s, color, (r, 0, w-2*r, h))
    pygame.draw.rect(s, color, (0, r, w, h-2*r))
    pygame.draw.circle(s, color, (r, r), r)
    pygame.draw.circle(s, color, (w-r, r), r)
    pygame.draw.circle(s, color, (r, h-r), r)
    pygame.draw.circle(s, color, (w-r, h-r), r)
    surf.blit(s, (x, y))
    if border > 0:
        pygame.draw.rect(surf, border_color, (x, y, w, h), border, border_radius=radius)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mechanics",
        default=MECH_CSV,
        help="Path to mechanics CSV (item_id,a,b,difficulty,speedup,releaseInterval,levelUpHits)"
    )
    args = ap.parse_args()

    # ---------- pygame init ----------
    pygame.init()
    info = pygame.display.Info()
    WIDTH, HEIGHT = info.current_w, info.current_h
    window = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)
    pygame.display.set_caption("QuickCalc 🎈")
    clock = pygame.time.Clock()

    # ---------- ui manager ----------
    ui_manager = pygame_gui.UIManager((WIDTH, HEIGHT))

    # ---------- layout ----------
    # Left 4/5 = game area; Right 1/5 = UI sidebar
    sidebar_w = WIDTH // 5
    game_w = WIDTH - sidebar_w
    game_x = 0
    sidebar_x = game_w

    # Top banner height
    banner_h = max(64, int(HEIGHT * 0.08))
    banner_rect = pygame.Rect(0, 0, WIDTH, banner_h)

    # Game playfield area
    margin = int(game_w * 0.04)
    inner_w = game_w - 2 * margin
    channel_w = inner_w // 4
    channel_h = HEIGHT - banner_h - int(HEIGHT * 0.06)  # leave some bottom margin
    top_y = banner_h + int(HEIGHT * 0.02)

    # Fonts
    balloon_font = pygame.font.SysFont(None, max(22, int(channel_w * 0.12)))
    hud_font     = pygame.font.SysFont(None, 28)
    banner_font  = pygame.font.SysFont(None, 36)

    # ---------- engine ----------
    mechanics_df = load_mechanics(args.mechanics)
    engine = QuickCalcEngine(mechanics_df)

    # ---------- lives / score tracking ----------
    lives  = NUM_LIVES
    score  = 0
    prev_missed = 0
    prev_correct = 0

    # Input field + keypad at bottom of sidebar
    # --- UI sidebar (pygame_gui) ---

    # ---------- UI sidebar (no panel; draw our own light card) ----------
    BOTTOM_PAD = 50  # ← y padding from bottom

    field_h = 48
    keypad_height = int(HEIGHT * 0.42)
    keypad_top = HEIGHT - keypad_height - BOTTOM_PAD

    # positions are absolute in the window (no container)
    # draw a rounded "card" behind the keypad in the draw loop (see below)

    input_field = pygame_gui.elements.UITextEntryLine(
        relative_rect=pygame.Rect(sidebar_x + 12, keypad_top, sidebar_w - 24, field_h),
        manager=ui_manager
    )
    input_field.focus()
    input_field.set_text("")
    input_field.set_allowed_characters('numbers')

    row_gap = 8
    btn_gap = 8
    btn_h = 48
    grid_top = keypad_top + field_h + row_gap

    clear_btn = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect(sidebar_x + 12, grid_top, (sidebar_w - 24 - btn_gap)//2, btn_h),
        text='Clear',
        manager=ui_manager
    )
    enter_btn = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect(sidebar_x + 12 + (sidebar_w - 24 - btn_gap)//2 + btn_gap,
                                grid_top, (sidebar_w - 24 - btn_gap)//2, btn_h),
        text='Enter',
        manager=ui_manager
    )

    grid_top2 = grid_top + btn_h + row_gap
    cols, rows = 3, 4
    cell_w = (sidebar_w - 24 - (cols - 1) * btn_gap) // cols
    cell_h = (keypad_top + keypad_height - grid_top2 - (rows - 1) * btn_gap)
    cell_h = max(44, cell_h // rows)

    labels = [["7","8","9"], ["4","5","6"], ["1","2","3"], ["0",".","⌫"]]
    key_buttons = []
    for r in range(rows):
        for c in range(cols):
            lx = sidebar_x + 12 + c * (cell_w + btn_gap)
            ly = grid_top2 + r * (cell_h + btn_gap)
            b = pygame_gui.elements.UIButton(
                relative_rect=pygame.Rect(lx, ly, cell_w, cell_h),
                text=labels[r][c],
                manager=ui_manager
            )
            key_buttons.append(b)


    # Exit button (top-right of banner)
    exit_btn = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect(WIDTH - 80, 12, 68, banner_h - 24),
        text='Exit',
        manager=ui_manager
    )

    # ---------- answer buffer ----------
    answer_buf = []  # list of chars

    running = True
    while running:
        dt = clock.tick(60) / 1000.0

        # ---------- score / lives bookkeeping ----------
        if engine.correct_count != prev_correct:
            delta = engine.correct_count - prev_correct
            score += max(0, delta)
            prev_correct = engine.correct_count

        if engine.missed_count != prev_missed:
            delta = engine.missed_count - prev_missed
            lives -= max(0, delta)
            prev_missed = engine.missed_count
            if lives <= 0:
                print(f"Final score: {score}")
                running = False

        # ---------- EVENT LOOP ----------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print(f"Final score: {score}")
                running = False

            # Keyboard (UITextEntryLine handles its own typing)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    print(f"Final score: {score}")
                    running = False

                elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    text = input_field.get_text().strip()
                    if text:
                        engine.submit_answer(text)
                        input_field.set_text("")

            # On-screen keypad + buttons
            if event.type == pygame_gui.UI_BUTTON_PRESSED:
                if event.ui_element == exit_btn:
                    print(f"Final score: {score}")
                    running = False

                elif event.ui_element == enter_btn:
                    text = input_field.get_text().strip()
                    if text:
                        engine.submit_answer(text)
                        input_field.set_text("")

                elif event.ui_element == clear_btn:
                    input_field.set_text("")

                elif event.ui_element in key_buttons:
                    label = event.ui_element.text
                    if label == "⌫":
                        cur = input_field.get_text()
                        if cur:
                            input_field.set_text(cur[:-1])
                    elif label == ".":
                        pass  # ignore decimals
                    else:
                        input_field.set_text(input_field.get_text() + label)

            # let pygame_gui see every event
            ui_manager.process_events(event)

        # ---------- UPDATE ----------
        engine.update(dt)
        ui_manager.update(dt)

        # ---------- DRAW ----------
        window.fill((255, 255, 255))

        # Top banner
        pygame.draw.rect(window, (245, 245, 245), banner_rect)

        # Lives / Level / Score
        lives_txt = hud_font.render(f"Lives: {lives}", True, (30, 30, 30))
        window.blit(lives_txt, (16, banner_rect.centery - lives_txt.get_height() // 2))
        lvl_txt = banner_font.render(f"Level {engine.level}", True, (120, 120, 120))
        window.blit(lvl_txt, lvl_txt.get_rect(center=(WIDTH // 2, banner_rect.centery)))
        score_txt = hud_font.render(f"Score: {score}", True, (30, 30, 30))
        window.blit(score_txt, (WIDTH - sidebar_w - 16 - score_txt.get_width(),
                                banner_rect.centery - score_txt.get_height() // 2))

        # Channel frames
        for i in range(4):
            x_left = game_x + margin + i * channel_w
            pygame.draw.rect(window, (225, 225, 225),
                             (x_left, top_y, channel_w - 8, channel_h), width=2)

        # Balloons
        for b in list(engine.balloons):
            x_left = game_x + margin + b.channel * channel_w
            draw_balloon(window, engine, b, x_left, channel_w, top_y, channel_h, balloon_font)

        # Sidebar card + UI
        card_rect = pygame.Rect(sidebar_x + 6, keypad_top - 12, sidebar_w - 12, keypad_height + 24)
        draw_round_rect(window, card_rect, (245, 245, 245), radius=18)

        ui_manager.draw_ui(window)
        pygame.display.flip()



if __name__ == "__main__":
    main()
