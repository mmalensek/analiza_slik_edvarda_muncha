import { defineConfig } from 'vite'
import path from 'path'
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import fs from 'fs'


function figmaAssetResolver() {
  return {
    name: 'figma-asset-resolver',
    resolveId(id) {
      if (id.startsWith('figma:asset/')) {
        const filename = id.replace('figma:asset/', '')
        return path.resolve(__dirname, 'src/assets', filename)
      }
    },
  }
}

export default defineConfig({
  plugins: [
    figmaAssetResolver(),
    // The React and Tailwind plugins are both required for Make, even if
    // Tailwind is not being actively used – do not remove them
    react(),
    tailwindcss(),
  ],
  resolve: {
    alias: {
      // Alias @ to the src directory
      '@': path.resolve(__dirname, './src'),
    },
  },

  // File types to support raw imports. Never add .css, .tsx, or .ts files to this.
  assetsInclude: ['**/*.svg', '**/*.csv', '**/*.jpg', '**/*.jpeg', '**/*.png'],

  server: {
    middlewares: [
      (req, res, next) => {
        // Serve munch_paintings from the parent directory
        if (req.url.startsWith('/munch_paintings/')) {
          const filename = req.url.replace('/munch_paintings/', '')
          const filepath = path.join(__dirname, '../../munch_paintings', filename)
          
          if (fs.existsSync(filepath)) {
            const ext = path.extname(filepath).toLowerCase()
            const mimeTypes: { [key: string]: string } = {
              '.jpg': 'image/jpeg',
              '.jpeg': 'image/jpeg',
              '.png': 'image/png',
              '.gif': 'image/gif',
              '.webp': 'image/webp'
            }
            res.setHeader('Content-Type', mimeTypes[ext] || 'application/octet-stream')
            res.end(fs.readFileSync(filepath))
            return
          }
        }
        next()
      }
    ]
  }
})
