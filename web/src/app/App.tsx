import {
  useEffect,
  useState,
  useCallback,
  useMemo,
  useRef
} from 'react';

import Papa from 'papaparse';
import { motion, AnimatePresence } from 'motion/react';

import {
  ChevronLeft,
  ChevronRight,
  Calendar,
  MapPin,
  Ruler
} from 'lucide-react';

interface RawPainting {
  number: string;
  name: string;
  year: string;
  location: string;
  status: string;
  technique: string;
  size: string;
  filename: string;
}

interface Painting {
  title: string;
  year: string;
  medium: string;
  dimensions: string;
  location: string;
  description: string;
  image: string;
}

interface GroupedPainting extends Painting {
  originalIndex: number;
}

type PageView = 'timeline' | 'analytics';

type AnalysisView =
  | 'original'
  | 'edges'
  | 'saliency'
  | 'composition';

const availableAnalysisModes: AnalysisView[] = [
  'original',
  'edges',
  'saliency',
  'composition'
];

const analyticsCards: Exclude<
  AnalysisView,
  'original'
>[] = ['edges', 'saliency', 'composition'];

const analysisDescriptions: Record<
  AnalysisView,
  string
> = {
  original: 'Original painting view.',
  edges:
    'Edge magnitude map with detected structural lines.',
  saliency:
    'Visual saliency heatmap showing high-attention regions.',
  composition:
    'Composition balance and spatial density analysis.'
};

const timelineChartCards = [
  {
    title: 'Temporal Colour Evolution',
    description:
      'Tracks brightness, warmth, saturation and colour complexity across years.',
    src: '/generirani_grafi/timeline/color_trends.png'
  },
  {
    title: 'Texture Density',
    description:
      'Shows changes in gradient strength, edge density and Laplacian variance through time.',
    src: '/generirani_grafi/timeline/texture_density.png'
  },
  {
    title: 'Line Structure',
    description:
      'Follows the number, length and support of detected line structures over time.',
    src: '/generirani_grafi/timeline/line_structure.png'
  },
  {
    title: 'Curves & Orientation',
    description:
      'Compares curvature and orientation behaviour across Munch’s career.',
    src: '/generirani_grafi/timeline/curve_structure.png'
  },
  {
    title: 'Symmetry & Composition',
    description:
      'Adds long-term symmetry and composition balance trends as a dedicated graph.',
    src: '/generirani_grafi/timeline/symmetry_composition.png'
  }
];


export default function App() {
  const [paintings, setPaintings] = useState<
    Painting[]
  >([]);
  const [selectedIndex, setSelectedIndex] =
    useState(0);
  const [expandedYear, setExpandedYear] =
    useState<string | null>(null);
  const [isLoading, setIsLoading] =
    useState(true);

  const [pageView, setPageView] =
    useState<PageView>('timeline');

  const [analysisView, setAnalysisView] =
    useState<AnalysisView>('original');

  const [showPalette, setShowPalette] =
    useState(false);

  const timelineRef =
    useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    fetch('/data/edvard_munch_cleaned.csv')
      .then((response) => response.text())
      .then((csv) => {
        Papa.parse(csv, {
          header: true,
          skipEmptyLines: true,
          complete: (results) => {
            const mapped = (
              results.data as RawPainting[]
            )
              .filter(
                (p) =>
                  p.filename &&
                  p.filename.trim() !== ''
              )
              .map((raw) => ({
                title: raw.name || 'Unknown',
                year: raw.year || 'Unknown',
                medium:
                  raw.technique || 'Unknown',
                dimensions: raw.size || 'Unknown',
                location:
                  raw.location || 'Unknown',
                description: raw.status
                  ? `Status: ${raw.status}`
                  : 'No description available',
                image:
                  raw.filename ||
                  'placeholder.jpg'
              }));

            const sorted = mapped.sort(
              (a, b) =>
                parseInt(a.year) -
                parseInt(b.year)
            );

            setPaintings(sorted);

            if (sorted.length > 0) {
              setExpandedYear(sorted[0].year);
            }

            setIsLoading(false);
          }
        });
      });
  }, []);

  const paintingsByYear = useMemo(() => {
    return paintings.reduce((acc, painting, index) => {
      if (!acc[painting.year]) {
        acc[painting.year] = [];
      }

      acc[painting.year].push({
        ...painting,
        originalIndex: index
      });

      return acc;
    }, {} as Record<string, GroupedPainting[]>);
  }, [paintings]);

  const timelineYears = useMemo(() => {
    return Object.keys(paintingsByYear).sort(
      (a, b) => Number(a) - Number(b)
    );
  }, [paintingsByYear]);

  const selectedPainting =
    paintings[selectedIndex];

  const selectedPaintingFolder = useMemo(() => {
    if (!selectedPainting) return '';
    return selectedPainting.image.split('.')[0];
  }, [selectedPainting]);

  const getMainVisualSrc = useCallback(() => {
    if (!selectedPainting) return '';

    if (showPalette) {
      return `/generirani_grafi/${selectedPaintingFolder}/palette.png`;
    }

    if (analysisView === 'original') {
      return `/munch_paintings/${selectedPainting.image}`;
    }

    return `/generirani_grafi/${selectedPaintingFolder}/${analysisView}.png`;
  }, [
    selectedPainting,
    selectedPaintingFolder,
    showPalette,
    analysisView
  ]);

  const handlePrevious = useCallback(() => {
    setShowPalette(false);
    setSelectedIndex((prev) =>
      prev > 0
        ? prev - 1
        : paintings.length - 1
    );
  }, [paintings.length]);

  const handleNext = useCallback(() => {
    setShowPalette(false);
    setSelectedIndex((prev) =>
      prev < paintings.length - 1
        ? prev + 1
        : 0
    );
  }, [paintings.length]);

  useEffect(() => {
    const handleKeyDown = (
      e: KeyboardEvent
    ) => {
      if (e.key === 'ArrowLeft') {
        handlePrevious();
      }

      if (e.key === 'ArrowRight') {
        handleNext();
      }
    };

    window.addEventListener(
      'keydown',
      handleKeyDown
    );

    return () =>
      window.removeEventListener(
        'keydown',
        handleKeyDown
      );
  }, [handlePrevious, handleNext]);

  const selectYear = (year: string) => {
    setExpandedYear(year);
    setShowPalette(false);

    const firstPainting =
      paintingsByYear[year]?.[0];

    if (firstPainting) {
      setSelectedIndex(
        firstPainting.originalIndex
      );
    }
  };

  const scrollTimelineLeft = () => {
    timelineRef.current?.scrollBy({
      left: -500,
      behavior: 'smooth'
    });
  };

  const scrollTimelineRight = () => {
    timelineRef.current?.scrollBy({
      left: 500,
      behavior: 'smooth'
    });
  };

  useEffect(() => {
    const timeline = timelineRef.current;

    if (!timeline) return;

    const handleWheel = (e: WheelEvent) => {
      if (
        Math.abs(e.deltaY) >
        Math.abs(e.deltaX)
      ) {
        e.preventDefault();
        timeline.scrollLeft += e.deltaY;
      }
    };

    timeline.addEventListener(
      'wheel',
      handleWheel,
      {
        passive: false
      }
    );

    return () => {
      timeline.removeEventListener(
        'wheel',
        handleWheel
      );
    };
  }, []);

  if (isLoading) {
    return (
      <div className="min-h-screen bg-[#0f1115] flex items-center justify-center text-white">
        Loading Munch&apos;s masterpieces...
      </div>
    );
  }

  if (!selectedPainting) {
    return (
      <div className="min-h-screen bg-[#0f1115] flex items-center justify-center text-white">
        No paintings available.
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#0f1115] text-white overflow-hidden">
      <header className="border-b border-white/10">
        <div className="max-w-7xl mx-auto px-8 py-8">
          <h1 className="text-5xl font-light tracking-tight">
            Edvard Munch
          </h1>

          <p className="text-white/50 mt-2 tracking-wide">
            1863 — 1944 · Norwegian Painter
          </p>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-8 pt-8 flex gap-4">
        {['timeline', 'analytics'].map((view) => (
          <button
            key={view}
            onClick={() => {
              setPageView(view as PageView);
            }}
            className={`px-5 py-3 rounded-full border transition-all capitalize ${
              pageView === view
                ? 'bg-amber-400 text-black border-amber-400'
                : 'bg-white/5 border-white/10 text-white hover:bg-white/10'
            }`}
          >
            {view}
          </button>
        ))}
      </div>

      <section className="border-b border-white/10">
        <div className="max-w-5xl mx-auto px-8 py-24">
          <div className="max-w-3xl">
            <div className="text-sm uppercase tracking-[0.4em] text-amber-400 mb-6">
              Visual Analysis Project
            </div>

            <h2 className="text-6xl leading-tight font-light">
              Exploring Edvard Munch through
              colour, texture and form.
            </h2>

            <p className="mt-8 text-lg text-white/60 leading-relaxed">
              This interactive archive combines
              computer vision and art history
              to analyse Edvard Munch&apos;s
              paintings through dominant
              colours, saliency maps, edge
              structures, composition studies
              and temporal trends across his
              artistic career.
            </p>
          </div>
        </div>
      </section>

      {pageView === 'timeline' && (
        <>
          <main className="max-w-7xl mx-auto px-8 py-12">
            <AnimatePresence mode="wait">
              <motion.div
                key={`${selectedIndex}-${analysisView}-${showPalette}`}
                initial={{
                  opacity: 0,
                  y: 0
                }}
                animate={{
                  opacity: 1,
                  y: 0
                }}
                exit={{
                  opacity: 0,
                  y: 0
                }}
                transition={{
                  duration: 0.15
                }}
                className="grid lg:grid-cols-2 gap-16 items-center"
              >
                <div className="relative">
                  <div className="relative h-[72vh] w-full rounded-3xl border border-white/10 bg-[#11131a] overflow-hidden backdrop-blur-sm flex items-center justify-center p-8">
                  <AnimatePresence mode="wait" initial={false}>
                    <motion.img
                      key={`${selectedPainting.image}-${analysisView}`}
                      src={
                        analysisView === 'original'
                          ? `/munch_paintings/${selectedPainting.image}`
                          : `/generirani_grafi/${selectedPainting.image.split('.')[0]}/${analysisView}.png`
                      }
                      alt={selectedPainting.title}
                      initial={{
                        opacity: 0,
                        scale: 0.985,
                        filter: 'blur(6px)'
                      }}
                      animate={{
                        opacity: 1,
                        scale: 1,
                        filter: 'blur(0px)'
                      }}
                      exit={{
                        opacity: 0,
                        scale: 1.01,
                        filter: 'blur(4px)'
                      }}
                      transition={{
                        duration: 0.05,
                        ease: 'easeInOut'
                      }}
                      className={`absolute inset-0 m-auto max-h-full max-w-full shadow-2xl ${
                        analysisView === 'original'
                          ? 'object-contain'
                          : 'object-contain rounded-2xl'
                      }`}
                    />
                  </AnimatePresence>
                </div>


                  <div className="absolute top-6 left-6 flex gap-2 z-20 flex-wrap pr-6">
                    {availableAnalysisModes.map(
                      (mode) => (
                        <button
                          key={mode}
                          onClick={() => {
                            setShowPalette(false);
                            setAnalysisView(mode);
                          }}
                          className={`px-4 py-2 rounded-full text-sm backdrop-blur-md border transition-all ${
                            analysisView === mode &&
                            !showPalette
                              ? 'bg-amber-400 text-black border-amber-400'
                              : 'bg-black/40 text-white border-white/10 hover:bg-white/10'
                          }`}
                        >
                          {mode}
                        </button>
                      )
                    )}

                    <button
                      onClick={() =>
                        setShowPalette(
                          (prev) => !prev
                        )
                      }
                      className={`px-4 py-2 rounded-full text-sm backdrop-blur-md border transition-all ${
                        showPalette
                          ? 'bg-amber-400 text-black border-amber-400'
                          : 'bg-black/40 text-white border-white/10 hover:bg-white/10'
                      }`}
                    >
                      palette
                    </button>
                  </div>

                  {showPalette ? (
                    <motion.div
                      key={`${selectedPainting.image}-palette`}
                      initial={{
                        opacity: 0,
                        y: 10
                      }}
                      animate={{
                        opacity: 1,
                        y: 0
                      }}
                      transition={{
                        duration: 0.25
                      }}
                      className="mt-6 rounded-2xl border border-white/10 bg-white/[0.03] p-5"
                    >
                      <div className="text-sm uppercase tracking-[0.3em] text-amber-400 mb-3">
                        Palette
                      </div>

                      <img
                        src={`/generirani_grafi/${selectedPaintingFolder}/palette.png`}
                        alt={`${selectedPainting.title} palette`}
                        className="w-full rounded-2xl object-contain max-h-[28rem] bg-black/20"
                      />
                    </motion.div>
                  ) : (
                    <motion.div
                      key={analysisView}
                      initial={{
                        opacity: 0,
                        y: 10
                      }}
                      animate={{
                        opacity: 1,
                        y: 0
                      }}
                      transition={{
                        duration: 0.25
                      }}
                      className="mt-6 rounded-2xl border border-white/10 bg-white/[0.03] p-5"
                    >
                      <div className="text-sm uppercase tracking-[0.3em] text-amber-400 mb-3">
                        Analysis Description
                      </div>

                      <p className="text-white/60 leading-relaxed">
                        {
                          analysisDescriptions[
                            analysisView
                          ]
                        }
                      </p>
                    </motion.div>
                  )}
                </div>

                <div className="space-y-8">
                  <div>
                    <motion.h2
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      className="text-6xl font-light leading-tight tracking-tight"
                    >
                      {selectedPainting.title}
                    </motion.h2>

                    <div className="flex items-center gap-2 mt-5 text-amber-400">
                      <Calendar className="w-4 h-4" />

                      <span className="tracking-wider">
                        {selectedPainting.year}
                      </span>
                    </div>
                  </div>

                  <div className="space-y-6 text-white/60">
                    <div className="flex gap-4">
                      <Ruler className="w-5 h-5 mt-1 flex-shrink-0 text-white/40" />

                      <div>
                        <div className="text-white/90 uppercase tracking-wider text-sm mb-1">
                          Medium & Dimensions
                        </div>

                        <div>
                          {selectedPainting.medium}
                        </div>

                        <div>
                          {
                            selectedPainting.dimensions
                          }
                        </div>
                      </div>
                    </div>

                    <div className="flex gap-4">
                      <MapPin className="w-5 h-5 mt-1 flex-shrink-0 text-white/40" />

                      <div>
                        <div className="text-white/90 uppercase tracking-wider text-sm mb-1">
                          Current Location
                        </div>

                        <div>
                          {selectedPainting.location}
                        </div>
                      </div>
                    </div>

                    <div className="flex gap-3 justify-start pt-6">
                      <button
                        onClick={handlePrevious}
                        className="w-12 h-12 rounded-full bg-white/10 hover:bg-white/20 border border-white/10 backdrop-blur-md flex items-center justify-center transition-all"
                      >
                        <ChevronLeft className="w-5 h-5" />
                      </button>

                      <button
                        onClick={handleNext}
                        className="w-12 h-12 rounded-full bg-white/10 hover:bg-white/20 border border-white/10 backdrop-blur-md flex items-center justify-center transition-all"
                      >
                        <ChevronRight className="w-5 h-5" />
                      </button>
                    </div>
                  </div>
                </div>
              </motion.div>
            </AnimatePresence>
          </main>

          <section className="border-t border-white/10 bg-black/20">
            <div className="max-w-7xl mx-auto px-8 py-20">
              <div className="flex items-center gap-5 mb-20">
                <div className="text-xs uppercase tracking-[0.4em] text-white/40">
                  Timeline
                </div>

                <div className="flex-1 h-px bg-white/10" />
              </div>

              <div className="relative">
                <div className="absolute top-1/2 left-0 right-0 h-px bg-white/10" />

                <div className="absolute right-0 -top-16 flex gap-3 z-20">
                  <button
                    onClick={scrollTimelineLeft}
                    className="w-11 h-11 rounded-full bg-white/5 hover:bg-white/10 border border-white/10 backdrop-blur-md flex items-center justify-center transition-all"
                  >
                    <ChevronLeft className="w-4 h-4" />
                  </button>

                  <button
                    onClick={scrollTimelineRight}
                    className="w-11 h-11 rounded-full bg-white/5 hover:bg-white/10 border border-white/10 backdrop-blur-md flex items-center justify-center transition-all"
                  >
                    <ChevronRight className="w-4 h-4" />
                  </button>
                </div>

                <div
                  ref={timelineRef}
                  className="overflow-x-auto overflow-y-hidden no-scrollbar scroll-smooth"
                >
                  <div className="flex items-center gap-20 min-w-max px-8 py-10">
                    {timelineYears.map(
                      (year, idx) => {
                        const representative =
                          paintingsByYear[year][0];

                        const isActive =
                          expandedYear === year;

                        return (
                          <motion.button
                            key={year}
                            whileHover={{
                              scale: 1.04
                            }}
                            whileTap={{
                              scale: 0.96
                            }}
                            onClick={() =>
                              selectYear(year)
                            }
                            className="relative flex flex-col items-center group"
                          >
                            {idx % 2 === 0 && (
                              <div className="mb-8">
                                <div
                                  className={`w-28 h-28 rounded-full overflow-hidden border-[3px] transition-all duration-300 ${
                                    isActive
                                      ? 'border-amber-400 scale-110 shadow-[0_0_40px_rgba(251,191,36,0.35)]'
                                      : 'border-white/15 group-hover:border-white/40'
                                  }`}
                                >
                                  <img
                                    src={`/munch_paintings/${representative.image}`}
                                    alt={
                                      representative.title
                                    }
                                    className="w-full h-full object-cover"
                                  />
                                </div>
                              </div>
                            )}

                            <div
                              className={`w-5 h-5 rounded-full border-[5px] z-10 transition-all duration-300 ${
                                isActive
                                  ? 'bg-amber-400 border-amber-400 scale-125'
                                  : 'bg-[#0f1115] border-white/30'
                              }`}
                            />

                            <div
                              className={`mt-5 text-sm tracking-[0.25em] transition-all duration-300 ${
                                isActive
                                  ? 'text-amber-400'
                                  : 'text-white/50'
                              }`}
                            >
                              {year}
                            </div>

                            {idx % 2 === 1 && (
                              <div className="mt-8">
                                <div
                                  className={`w-28 h-28 rounded-full overflow-hidden border-[3px] transition-all duration-300 ${
                                    isActive
                                      ? 'border-amber-400 scale-110 shadow-[0_0_40px_rgba(251,191,36,0.35)]'
                                      : 'border-white/15 group-hover:border-white/40'
                                  }`}
                                >
                                  <img
                                    src={`/munch_paintings/${representative.image}`}
                                    alt={
                                      representative.title
                                    }
                                    className="w-full h-full object-cover"
                                  />
                                </div>
                              </div>
                            )}
                          </motion.button>
                        );
                      }
                    )}
                  </div>
                </div>
              </div>

              <AnimatePresence mode="wait">
                {expandedYear && (
                  <motion.div
                    key={expandedYear}
                    initial={{
                      opacity: 0,
                      y: 15
                    }}
                    animate={{
                      opacity: 1,
                      y: 0
                    }}
                    exit={{
                      opacity: 0,
                      y: 15
                    }}
                    transition={{
                      duration: 0.3
                    }}
                    className="mt-24"
                  >
                    <div className="flex items-center gap-5 mb-10">
                      <h3 className="text-4xl font-light text-amber-400">
                        {expandedYear}
                      </h3>

                      <div className="flex-1 h-px bg-white/10" />
                    </div>

                    <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-5">
                      {paintingsByYear[
                        expandedYear
                      ].map((painting) => {
                        const isSelected =
                          painting.originalIndex ===
                          selectedIndex;

                        return (
                          <motion.button
                            whileHover={{
                              y: -4
                            }}
                            key={
                              painting.originalIndex
                            }
                            onClick={() => {
                              setShowPalette(
                                false
                              );
                              setSelectedIndex(
                                painting.originalIndex
                              );
                            }}
                            className="group text-left"
                          >
                            <div
                              className={`aspect-[3/4] overflow-hidden rounded-2xl border transition-all duration-300 ${
                                isSelected
                                  ? 'border-amber-400 shadow-[0_0_30px_rgba(251,191,36,0.2)]'
                                  : 'border-white/10'
                              }`}
                            >
                              <img
                                src={`/munch_paintings/${painting.image}`}
                                alt={
                                  painting.title
                                }
                                className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500"
                              />
                            </div>

                            <div className="mt-3">
                              <div className="text-sm text-white/90 truncate">
                                {painting.title}
                              </div>

                              <div className="text-xs text-white/40 mt-1">
                                {painting.year}
                              </div>
                            </div>
                          </motion.button>
                        );
                      })}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </section>
        </>
      )}

      {pageView === 'analytics' && (
        <section className="max-w-7xl mx-auto px-8 py-20">
          <div className="mb-24">
            <div className="text-sm uppercase tracking-[0.4em] text-amber-400 mb-6">
              Computational Analysis
            </div>

            <h2 className="text-6xl font-light leading-tight max-w-5xl">
              Machine vision analysis of
              Edvard Munch&apos;s paintings.
            </h2>

            <p className="mt-8 text-white/50 text-lg leading-relaxed max-w-3xl">
              Each artwork is analysed through
              colour clustering, saliency,
              edge detection, composition
              structure and long-term temporal
              graphs.
            </p>
          </div>

          <div className="mb-24 rounded-[2rem] overflow-hidden border border-white/10 bg-black/30 p-8">
            <img
              src={getMainVisualSrc()}
              alt={
                showPalette
                  ? `${selectedPainting.title} palette`
                  : analysisView
              }
              className="w-full rounded-2xl object-contain max-h-[75vh]"
            />

            <div className="mt-6">
              <div className="text-sm uppercase tracking-[0.3em] text-amber-400 mb-3">
                {showPalette
                  ? 'Current Palette'
                  : 'Current Analysis'}
              </div>

              <h3 className="text-3xl font-light capitalize">
                {showPalette
                  ? 'palette'
                  : analysisView}
              </h3>

              <p className="mt-4 text-white/50 max-w-3xl">
                {showPalette
                  ? 'Dominant colour palette extracted for the selected painting.'
                  : analysisDescriptions[
                      analysisView
                    ]}
              </p>
            </div>
          </div>

          <div className="flex flex-wrap gap-3 mb-10">
            {availableAnalysisModes.map((mode) => (
              <button
                key={mode}
                onClick={() => {
                  setShowPalette(false);
                  setAnalysisView(mode);
                }}
                className={`px-4 py-2 rounded-full text-sm border transition-all capitalize ${
                  analysisView === mode &&
                  !showPalette
                    ? 'bg-amber-400 text-black border-amber-400'
                    : 'bg-white/5 border-white/10 text-white hover:bg-white/10'
                }`}
              >
                {mode}
              </button>
            ))}

            <button
              onClick={() =>
                setShowPalette((prev) => !prev)
              }
              className={`px-4 py-2 rounded-full text-sm border transition-all ${
                showPalette
                  ? 'bg-amber-400 text-black border-amber-400'
                  : 'bg-white/5 border-white/10 text-white hover:bg-white/10'
              }`}
            >
              palette
            </button>
          </div>

          <div className="grid md:grid-cols-2 xl:grid-cols-3 gap-8">
            {analyticsCards.map((mode) => (
              <motion.button
                whileHover={{ y: -5 }}
                key={mode}
                onClick={() => {
                  setShowPalette(false);
                  setAnalysisView(mode);
                }}
                className={`group rounded-[2rem] overflow-hidden border transition-all ${
                  analysisView === mode &&
                  !showPalette
                    ? 'border-amber-400'
                    : 'border-white/10 hover:border-white/30'
                }`}
              >
                <div className="aspect-[4/3] bg-black/40 overflow-hidden">
                  <img
                    src={`/generirani_grafi/${selectedPaintingFolder}/${mode}.png`}
                    alt={mode}
                    className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500"
                  />
                </div>

                <div className="p-6 text-left">
                  <div className="text-sm uppercase tracking-[0.25em] text-amber-400 mb-3">
                    Analysis
                  </div>

                  <h3 className="text-2xl font-light capitalize">
                    {mode}
                  </h3>

                  <p className="mt-4 text-white/50 text-sm leading-relaxed">
                    {
                      analysisDescriptions[
                        mode
                      ]
                    }
                  </p>
                </div>
              </motion.button>
            ))}

            <motion.button
              whileHover={{ y: -5 }}
              onClick={() =>
                setShowPalette(true)
              }
              className={`group rounded-[2rem] overflow-hidden border transition-all ${
                showPalette
                  ? 'border-amber-400'
                  : 'border-white/10 hover:border-white/30'
              }`}
            >
              <div className="aspect-[4/3] bg-black/40 overflow-hidden">
                <img
                  src={`/generirani_grafi/${selectedPaintingFolder}/palette.png`}
                  alt="palette"
                  className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500"
                />
              </div>

              <div className="p-6 text-left">
                <div className="text-sm uppercase tracking-[0.25em] text-amber-400 mb-3">
                  Analysis
                </div>

                <h3 className="text-2xl font-light capitalize">
                  palette
                </h3>

                <p className="mt-4 text-white/50 text-sm leading-relaxed">
                  Dominant colour palette for
                  the selected painting.
                </p>
              </div>
            </motion.button>
          </div>

          <div className="mt-24">
            <div className="flex items-center gap-5 mb-10">
              <div className="text-sm uppercase tracking-[0.4em] text-amber-400">
                Timeline Metrics
              </div>

              <div className="flex-1 h-px bg-white/10" />
            </div>

            <div className="grid grid-cols-1 gap-10">
              {timelineChartCards.map((chart) => (
                <div
                  key={chart.title}
                  className="rounded-[2rem] overflow-hidden border border-white/10 bg-black/30 p-8"
                >
                  <div className="aspect-[16/8] overflow-hidden rounded-2xl bg-[#0b0d12] border border-white/5">
                    <img
                      src={chart.src}
                      alt={chart.title}
                      className="w-full h-full object-contain"
                    />
                  </div>

                  <div className="mt-6 max-w-4xl">
                    <h3 className="text-3xl font-light">
                      {chart.title}
                    </h3>

                    <p className="mt-3 text-base text-white/50 leading-relaxed">
                      {chart.description}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </div>

        </section>
      )}
    </div>
  );
}
