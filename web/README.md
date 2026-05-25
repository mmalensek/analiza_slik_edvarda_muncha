# Edvard Munch Gallery

A sleek, interactive timeline website showcasing the paintings of Edvard Munch (1863-1944).

## Features

- **Interactive Timeline**: Horizontal scrollable timeline showing all paintings chronologically
- **Smooth Animations**: Elegant transitions powered by Motion (Framer Motion)
- **Keyboard Navigation**: Use arrow keys (← →) to navigate through paintings
- **Responsive Design**: Works beautifully on desktop and mobile devices
- **Rich Details**: View painting information including year, medium, dimensions, and location

## Data Structure

The gallery reads from `/public/data/edvard_munch.csv` with the following columns:

- `title`: Painting title
- `year`: Year created
- `medium`: Artistic medium used
- `dimensions`: Physical dimensions
- `location`: Current location/museum
- `description`: Brief description of the work
- `image`: Image filename (should be placed in `/public/munch_paintings/`)

## Adding Real Images

To replace placeholder images with actual Munch paintings:

1. Download high-quality images of Munch's paintings
2. Name them according to the `image` column in the CSV (e.g., `the-scream.jpg`, `madonna.jpg`)
3. Place them in the `/public/munch_paintings/` directory

The application currently uses Unsplash art images as fallbacks when actual images aren't found.

## Run the App

### Frontend

Run the website from the `web` folder:

```bash
pnpm install
pnpm dev
```

### Backend

```bash
# samo zaženi skripto server
python3 web/api/server.py
```


The app should open at the local Vite URL shown in the terminal, usually `http://localhost:5173`.

## Navigation

- **Click** on any thumbnail in the timeline to jump to that painting
- **Arrow Buttons**: Click the left/right arrows on the main display
- **Keyboard**: Press ← or → arrow keys to navigate
- **Timeline Scroll**: Scroll horizontally through the timeline at the bottom

## Technology Stack

- React 18
- TypeScript
- Tailwind CSS v4
- Motion (Framer Motion)
- PapaParse (CSV parsing)
- Lucide React (icons)
