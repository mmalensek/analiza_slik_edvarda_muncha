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

export default function App() {
  const [paintings, setPaintings] = useState<Painting[]>([]);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [expandedYear, setExpandedYear] =
    useState<string | null>(null);

  const [isLoading, setIsLoading] = useState(true);

  const [pageView, setPageView] =
    useState<'timeline' | 'analytics'>(
      'timeline'
  );

  const [analysisView, setAnalysisView] =
    useState<
      'original' |
      'edges' |
      'saliency'
    >('original');

  const [showPalette, setShowPalette] =
    useState(false);

  const timelineRef =
    useRef<HTMLDivElement | null>(null);

  // =========================
  // LOAD CSV
  // =========================

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

  // =========================
  // GROUP BY YEAR
  // =========================

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

  // =========================
  // NAVIGATION
  // =========================

  const handlePrevious = useCallback(() => {
    setSelectedIndex((prev) =>
      prev > 0
        ? prev - 1
        : paintings.length - 1
    );
  }, [paintings.length]);

  const handleNext = useCallback(() => {
    setSelectedIndex((prev) =>
      prev < paintings.length - 1
        ? prev + 1
        : 0
    );
  }, [paintings.length]);

  // keyboard arrows

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

  // =========================
  // SELECT YEAR
  // =========================

  const selectYear = (year: string) => {
    setExpandedYear(year);

    const firstPainting =
      paintingsByYear[year]?.[0];

    if (firstPainting) {
      setSelectedIndex(
        firstPainting.originalIndex
      );
    }
  };

  // =========================
  // TIMELINE SCROLL
  // =========================

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

  // horizontal wheel scroll

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

  // =========================
  // LOADING
  // =========================

  if (isLoading) {
    return (
      <div className="min-h-screen bg-[#0f1115] flex items-center justify-center text-white">
        Loading Munch's masterpieces...
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#0f1115] text-white overflow-hidden">
      {/* ========================= */}
      {/* HEADER */}
      {/* ========================= */}

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

      {/* ========================= */}
      {/* PAGE SWITCHER */}
      {/* ========================= */}

      <div className="max-w-7xl mx-auto px-8 pt-8 flex gap-4">
        {['timeline', 'analytics'].map((view) => (
          <button
            key={view}
            onClick={() =>
              setPageView(
                view as typeof pageView
              )
            }
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
              to analyse Edvard Munch's paintings
              through dominant colours,
              saliency maps, texture metrics,
              edge structures and temporal trends
              across his artistic career.
            </p>
          </div>
        </div>
      </section>

      {/* ========================= */}
      {/* MAIN SECTION */}
      {/* ========================= */}

      {pageView === 'timeline' && (
        <>

      <main className="max-w-7xl mx-auto px-8 py-12">
        <AnimatePresence mode="wait">
          {selectedPainting && (
            <motion.div
              key={selectedIndex}
              initial={{
                opacity: 0,
                y: 10
              }}
              animate={{
                opacity: 1,
                y: 0
              }}
              exit={{
                opacity: 0,
                y: 10
              }}
              transition={{
                duration: 0.35
              }}
              className="grid lg:grid-cols-2 gap-16 items-center"
            >
              {/* ========================= */}
              {/* IMAGE */}
              {/* ========================= */}

              <div className="relative">
                {/* FIXED IMAGE CONTAINER */}
                <div className="h-[72vh] w-full rounded-3xl border border-white/10 bg-black/30 overflow-hidden backdrop-blur-sm flex items-center justify-center p-8">
                 <img
                  src={
                    analysisView === 'original'
                      ? `/munch_paintings/${selectedPainting.image}`
                      : `/generirani_grafi/${
                          selectedPainting.image.split('.')[0]
                        }/${analysisView}.png`
                  }
                  alt={selectedPainting.title}
                  className="max-h-full max-w-full object-contain shadow-2xl transition-all duration-500"
                />

                </div>

                <div className="absolute top-6 left-6 flex gap-2 z-20">
                  {[
                    'original',
                    'edges',
                    'saliency'
                  ].map((mode) => (
                    <button
                      key={mode}
                      onClick={() =>
                        setAnalysisView(
                          mode as typeof analysisView
                        )
                      }
                      className={`px-4 py-2 rounded-full text-sm backdrop-blur-md border transition-all ${
                        analysisView === mode
                          ? 'bg-amber-400 text-black border-amber-400'
                          : 'bg-black/40 text-white border-white/10 hover:bg-white/10'
                      }`}
                    >
                      {mode}
                    </button>
                  ))}

                  <button
                    onClick={() => {
                      setAnalysisView('original');
                      setShowPalette((prev) => !prev);
                    }}
                    className={`px-4 py-2 rounded-full text-sm backdrop-blur-md border transition-all ${
                      showPalette
                        ? 'bg-amber-400 text-black border-amber-400'
                        : 'bg-black/40 text-white border-white/10 hover:bg-white/10'
                    }`}
                  >
                    palette
                  </button>

                </div>

                  <AnimatePresence>
                    {showPalette && (
                      <motion.div
                        initial={{ opacity: 0, y: -8, scale: 0.98 }}
                        animate={{ opacity: 1, y: 0, scale: 1 }}
                        exit={{ opacity: 0, y: -8, scale: 0.98 }}
                        transition={{ duration: 0.2 }}
                        className="absolute top-20 left-6 z-30 w-64 rounded-2xl border border-white/10 bg-black/75 backdrop-blur-xl p-4 shadow-2xl"
                      >
                        <div className="text-xs uppercase tracking-[0.25em] text-white/50 mb-3">
                          Colour Palette
                        </div>

                        <img
                          src={`/generirani_grafi/${
                            selectedPainting.image.split('.')[0]
                          }/palette.png`}
                          alt={`${selectedPainting.title} colour palette`}
                          className="w-full rounded-xl object-contain"
                        />
                      </motion.div>
                    )}
                  </AnimatePresence>


                {/* ARROWS */}
                <div className="absolute bottom-6 left-6 flex gap-3">
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

              {/* ========================= */}
              {/* DETAILS */}
              {/* ========================= */}

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
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      {/* ========================= */}
      {/* TIMELINE */}
      {/* ========================= */}

      <section className="border-t border-white/10 bg-black/20">
        <div className="max-w-7xl mx-auto px-8 py-20">
          {/* TITLE */}
          <div className="flex items-center gap-5 mb-20">
            <div className="text-xs uppercase tracking-[0.4em] text-white/40">
              Timeline
            </div>

            <div className="flex-1 h-px bg-white/10" />
          </div>

          {/* TIMELINE CONTAINER */}
          <div className="relative">
            {/* LINE */}
            <div className="absolute top-1/2 left-0 right-0 h-px bg-white/10" />

            {/* NAVIGATION BUTTONS */}
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

            {/* HIDDEN SCROLLBAR */}
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
                        {/* TOP IMAGE */}
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

                        {/* DOT */}
                        <div
                          className={`w-5 h-5 rounded-full border-[5px] z-10 transition-all duration-300 ${
                            isActive
                              ? 'bg-amber-400 border-amber-400 scale-125'
                              : 'bg-[#0f1115] border-white/30'
                          }`}
                        />

                        {/* YEAR */}
                        <div
                          className={`mt-5 text-sm tracking-[0.25em] transition-all duration-300 ${
                            isActive
                              ? 'text-amber-400'
                              : 'text-white/50'
                          }`}
                        >
                          {year}
                        </div>

                        {/* BOTTOM IMAGE */}
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

          {/* ========================= */}
          {/* EXPANDED YEAR */}
          {/* ========================= */}

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
                {/* HEADER */}
                <div className="flex items-center gap-5 mb-10">
                  <h3 className="text-4xl font-light text-amber-400">
                    {expandedYear}
                  </h3>

                  <div className="flex-1 h-px bg-white/10" />
                </div>

                {/* GRID */}
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
                        onClick={() =>
                          setSelectedIndex(
                            painting.originalIndex
                          )
                        }
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
                            alt={painting.title}
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

        {/* ========================= */}
        {/* HERO */}
        {/* ========================= */}

        <div className="mb-24">
          <div className="text-sm uppercase tracking-[0.4em] text-amber-400 mb-6">
            Computational Analysis
          </div>

          <h2 className="text-6xl font-light leading-tight max-w-4xl">
            Temporal patterns across
            Edvard Munch’s artistic career.
          </h2>

          <p className="mt-8 text-white/50 text-lg leading-relaxed max-w-3xl">
            Using computer vision techniques,
            the collection was analysed through
            colour distribution, edge density,
            texture structure, saliency,
            line orientation and visual complexity
            to uncover long-term stylistic evolution.
          </p>
        </div>

        {/* ========================= */}
        {/* METRIC CARDS */}
        {/* ========================= */}

        <div className="grid md:grid-cols-3 gap-6 mb-24">

          {[
            {
              title: '1500+',
              subtitle: 'Paintings Analysed'
            },
            {
              title: '11',
              subtitle: 'Texture Metrics'
            },
            {
              title: '1860–1944',
              subtitle: 'Career Span'
            }
          ].map((item) => (
            <div
              key={item.title}
              className="rounded-3xl border border-white/10 bg-white/[0.03] p-8 backdrop-blur-sm"
            >
              <div className="text-5xl font-light text-amber-400">
                {item.title}
              </div>

              <div className="mt-4 text-white/50 tracking-wide">
                {item.subtitle}
              </div>
            </div>
          ))}
        </div>

        {/* ========================= */}
        {/* LARGE FEATURE GRAPH */}
        {/* ========================= */}

        <div className="mb-24">
          <div className="mb-8">
            <h3 className="text-3xl font-light">
              Colour Evolution Through Time
            </h3>

            <p className="text-white/50 mt-3 max-w-2xl">
              Analysis of brightness,
              warmth, saturation and
              chromatic complexity across
              different periods of Munch’s work.
            </p>
          </div>

          <div className="rounded-[2rem] overflow-hidden border border-white/10 bg-black/30 p-8">
            <img
              src="/generirani_grafi/timeline/color_trends.png"
              className="w-full rounded-2xl"
            />
          </div>
        </div>

        {/* ========================= */}
        {/* TWO COLUMN ANALYSIS */}
        {/* ========================= */}

        <div className="grid xl:grid-cols-2 gap-10 mb-24">

          {/* TEXTURE */}
          <div className="rounded-[2rem] border border-white/10 bg-white/[0.03] p-8">

            <div className="mb-6">
              <div className="text-sm uppercase tracking-[0.3em] text-amber-400 mb-3">
                Texture Analysis
              </div>

              <h3 className="text-3xl font-light">
                Density & Surface Complexity
              </h3>
            </div>

            <img
              src="/generirani_grafi/timeline/texture_density.png"
              className="w-full rounded-2xl"
            />
          </div>

          {/* LINES */}
          <div className="rounded-[2rem] border border-white/10 bg-white/[0.03] p-8">

            <div className="mb-6">
              <div className="text-sm uppercase tracking-[0.3em] text-amber-400 mb-3">
                Structural Analysis
              </div>

              <h3 className="text-3xl font-light">
                Straight Line Structures
              </h3>
            </div>

            <img
              src="/generirani_grafi/timeline/line_structure.png"
              className="w-full rounded-2xl"
            />
          </div>
        </div>

        {/* ========================= */}
        {/* FINAL LARGE GRAPH */}
        {/* ========================= */}

        <div className="mb-12">

          <div className="mb-8">
            <div className="text-sm uppercase tracking-[0.3em] text-amber-400 mb-3">
              Orientation Analysis
            </div>

            <h3 className="text-3xl font-light">
              Curvature & Directional Behaviour
            </h3>

            <p className="text-white/50 mt-3 max-w-2xl">
              Measuring orientation entropy,
              curvature and dominant directional
              structures within Munch’s compositions.
            </p>
          </div>

          <div className="rounded-[2rem] overflow-hidden border border-white/10 bg-black/30 p-8">
            <img
              src="/generirani_grafi/timeline/curve_structure.png"
              className="w-full rounded-2xl"
            />
          </div>
        </div>

      </section>
    )}
    </div>
  );
}