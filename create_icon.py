"""
Erstellt ein Icon für LagerPilot
Führe dieses Script einmal aus, um icon.ico zu generieren.
"""

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("Pillow wird installiert...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'Pillow'])
    from PIL import Image, ImageDraw, ImageFont

def create_lagerpilot_icon():
    """Erstellt ein modernes Icon für LagerPilot."""
    
    # Größen für ICO-Datei (Windows benötigt mehrere Größen)
    sizes = [16, 32, 48, 64, 128, 256]
    images = []
    
    for size in sizes:
        # Erstelle ein neues Bild mit transparentem Hintergrund
        img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Hintergrund: Abgerundetes Quadrat in Blau
        margin = size // 10
        bg_color = (41, 128, 185)  # Schönes Blau
        
        # Zeichne abgerundeten Hintergrund
        draw.rounded_rectangle(
            [margin, margin, size - margin, size - margin],
            radius=size // 5,
            fill=bg_color
        )
        
        # Zeichne ein stilisiertes Lager/Regal-Symbol
        # Drei horizontale Linien (Regalböden)
        line_color = (255, 255, 255)  # Weiß
        line_width = max(1, size // 16)
        
        padding = size // 4
        shelf_height = (size - 2 * padding) // 4
        
        for i in range(3):
            y = padding + shelf_height * (i + 1)
            draw.line(
                [(padding, y), (size - padding, y)],
                fill=line_color,
                width=line_width
            )
        
        # Vertikale Stützen links und rechts
        draw.line(
            [(padding, padding + shelf_height), (padding, size - padding)],
            fill=line_color,
            width=line_width
        )
        draw.line(
            [(size - padding, padding + shelf_height), (size - padding, size - padding)],
            fill=line_color,
            width=line_width
        )
        
        # Kleine Boxen auf den Regalen
        box_color = (241, 196, 15)  # Gelb/Gold
        box_size = max(2, size // 10)
        
        # Box auf oberem Regal
        y1 = padding + shelf_height - box_size
        draw.rectangle(
            [padding + box_size, y1, padding + box_size * 3, y1 + box_size],
            fill=box_color
        )
        
        # Box auf mittlerem Regal
        y2 = padding + shelf_height * 2 - box_size
        draw.rectangle(
            [size - padding - box_size * 3, y2, size - padding - box_size, y2 + box_size],
            fill=box_color
        )
        
        # Box auf unterem Regal
        y3 = padding + shelf_height * 3 - box_size
        draw.rectangle(
            [padding + box_size * 2, y3, padding + box_size * 4, y3 + box_size],
            fill=box_color
        )
        
        images.append(img)
    
    # Speichere als ICO-Datei
    images[0].save(
        'icon.ico',
        format='ICO',
        sizes=[(s, s) for s in sizes],
        append_images=images[1:]
    )
    
    print("✅ icon.ico wurde erfolgreich erstellt!")
    print(f"   Enthaltene Größen: {sizes}")
    return True

if __name__ == '__main__':
    create_lagerpilot_icon()
