import React, { useState, useEffect, useCallback } from "react";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, AreaChart } from "recharts";
import { Search, SlidersHorizontal } from "lucide-react";

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload || !payload.length) return null;

  // Find the mean, hi, and lo values
  const mean = payload.find(p => p.dataKey === 'mean')?.value;
  const hi = payload.find(p => p.dataKey === 'hi')?.value;
  const lo = payload.find(p => p.dataKey === 'lo')?.value;

  // Check if this is the predicted year (2025) by looking at the sd value
  const isPrediction = payload[0]?.payload?.sd > 0;

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-3 shadow-lg">
      <p className="font-semibold text-gray-900 mb-2">Season {label}</p>
      <div className="mb-2">
        <p className="text-sm text-gray-600">
          {isPrediction ? "Predicted xwOBA:" : "Actual xwOBA:"}
        </p>
        <p className={`font-mono text-lg font-bold ${isPrediction ? "text-blue-600" : "text-green-600"}`}>
          {mean !== undefined ? mean.toFixed(3) : "N/A"}
        </p>
      </div>
      {isPrediction && hi !== undefined && lo !== undefined && (
        <div className="text-xs text-gray-500">
          <p>Confidence interval:</p>
          <p className="font-mono">
            {lo.toFixed(3)} - {hi.toFixed(3)}
          </p>
        </div>
      )}
      <div className="mt-2 text-xs text-gray-400">
        <p>
          • <strong>{isPrediction ? "Predicted" : "Historical"}:</strong>{" "}
          {isPrediction
            ? "Model's forecast with uncertainty"
            : "Actual performance"}
        </p>
      </div>
    </div>
  );
};



const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

export default function App() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [selected, setSelected] = useState(null); // {id,name}
  const [curve, setCurve] = useState([]); // [{season, mean, sd}]
  const [whatIf, setWhatIf] = useState({ k: 0, bb: 0, maxev: 0, gbfb: 0 });
  const [explanation, setExplanation] = useState("");
  const [loadingExplain, setLoadingExplain] = useState(false);

  // --- live search --------------------------------------------------
  useEffect(() => {
    if (query.length < 3) return;
    const controller = new AbortController();
    fetch(`${API_BASE}/players/search?q=${query}`, { signal: controller.signal })
      .then((r) => r.json())
      .then(setResults)
      .catch(() => {});
    return () => controller.abort();
  }, [query]);

  // --- fetch forecast for selected player ---------------------------
  const fetchForecast = useCallback((overrides = null) => {
    if (!selected) return;
    const url = `${API_BASE}/players/${selected.id}/forecast`;
    const opts = overrides
      ? {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(overrides),
        }
      : {};
    fetch(url, opts)
      .then((r) => r.json())
      .then((data) => setCurve(data.curve))
      .catch(() => setCurve([]));
  }, [selected]);

  // --- fetch explanation -------------------------------------------
  const fetchExplanation = () => {
    if (!selected) return;
    setLoadingExplain(true);
    fetch(`${API_BASE}/players/${selected.id}/explain`)
      .then((r) => r.json())
      .then((d) => setExplanation(d.text))
      .finally(() => setLoadingExplain(false));
  };

  // re‑fetch curve when player changes
  useEffect(() => {
    setCurve([]);
    if (selected) fetchForecast();
  }, [selected, fetchForecast]);

  // --- handle what‑if slider commit --------------------------------
  const handleWhatIfCommit = () => {
    fetchForecast({ overrides: whatIf });
  };

  // helper for +/- 2σ ribbon plotting
  const curveWithBounds = curve.map((d) => ({
    ...d,
    lo: d.mean - 2 * d.sd,
    hi: d.mean + 2 * d.sd,
  }));

  return (
    <div className="min-h-screen bg-gray-100 p-4 text-gray-900">
      <div className="max-w-5xl mx-auto space-y-6">
        {/* Site Introduction */}
        <Card>
          <CardHeader>
            <CardTitle>MLB Aging Curve Lab</CardTitle>
            <p className="text-xs text-gray-500">Created by Irura Nyiha</p>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-gray-600 mb-4">
              This tool uses machine learning to predict how MLB players' performance will change as they age. 
              The model analyzes historical data to forecast future xwOBA (expected weighted On-Base Average) 
              trajectories, helping evaluate player development and decline patterns.
            </p>
            <p className="text-sm text-gray-600">
              <strong>How it works:</strong> Search for a player, view their predicted aging curve, and use the 
              "what-if" sliders to explore how changes in their underlying skills might affect future performance.
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Search size={18} /> Player search
            </CardTitle>
          </CardHeader>
          <CardContent>
            <Input
              placeholder="Start typing a player name…"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
            {results.length > 0 && (
              <ul className="mt-2 divide-y border rounded bg-white max-h-60 overflow-auto">
                {results.map((p) => (
                  <li
                    key={p.id}
                    className="px-3 py-2 hover:bg-slate-100 cursor-pointer"
                    onClick={() => {
                      setSelected(p);
                      setResults([]);
                      setQuery(p.name);
                    }}
                  >
                    {p.name} <span className="text-sm text-gray-500">{p.team}</span>
                  </li>
                ))}
              </ul>
            )}
          </CardContent>
        </Card>

        {selected && (
          <>
            {/* Aging curve chart */}
            <Card>
              <CardHeader>
                <CardTitle>
                  xwOBA Aging Curve — {selected.name}
                </CardTitle>
              </CardHeader>
              <CardContent className="h-72">
                {curve.length > 0 ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={curveWithBounds} margin={{ left: 20, right: 20 }}>
                      <defs>
                        <linearGradient id="shade" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor="#8884d8" stopOpacity={0.6} />
                          <stop offset="100%" stopColor="#8884d8" stopOpacity={0} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="season" />
                      <YAxis domain={["dataMin-0.04", "dataMax+0.04"]} 
                       tickFormatter={(value) => value.toFixed(2)}/>
                      <Tooltip content={<CustomTooltip />}  />
                      
                      <Area
                        type="monotone"
                        dataKey="hi"
                        stroke="none"
                        fillOpacity={1}
                        fill="url(#shade)"
                      />
                      <Area
                        type="monotone"
                        dataKey="lo"
                        stroke="none"
                        fillOpacity={1}
                        fill="#fff"
                      />
                      <Line type="monotone" dataKey="mean" stroke="#8884d8" dot />
                    </AreaChart>
                  </ResponsiveContainer>
                ) : (
                  <p className="text-sm text-gray-500">Loading curve…</p>
                )}
              </CardContent>
            </Card>

            {/* xwOBA Explanation */}
            <Card>
              <CardHeader>
                <CardTitle>Understanding xwOBA</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <p className="text-sm text-gray-600">
                    <strong>What is xwOBA?</strong> Think of it like a "batting score" that measures how good a hitter is. 
                     The higher the number, the better the hitter.
                  </p>
                  <p className="text-sm text-gray-600">
                    <strong>Simple analogy:</strong> Imagine you're grading a student's essay. You don't just count how many 
                    words they wrote (like counting hits in baseball), but you evaluate the quality of their writing, 
                    their arguments, and their overall impact. xwOBA does the same thing for baseball - it measures the 
                    quality of a player's hitting, not just how often they get hits.
                  </p>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                    <div>
                      <h4 className="font-semibold mb-2">What it measures:</h4>
                      <ul className="list-disc list-inside space-y-1 text-gray-600">
                        <li>How hard they hit the ball (exit velocity)</li>
                        <li>What angle they hit it at (launch angle)</li>
                        <li>How often they walk vs. strike out</li>
                        <li>Overall offensive value</li>
                      </ul>
                    </div>
                    <div>
                      <h4 className="font-semibold mb-2">Score interpretation:</h4>
                      <ul className="list-disc list-inside space-y-1 text-gray-600">
                        <li><strong>0.400+:</strong> (elite hitter)</li>
                        <li><strong>0.350-0.400:</strong> (above average)</li>
                        <li><strong>0.320-0.350:</strong> (average)</li>
                        <li><strong>Below 0.320:</strong> (below average)</li>
                      </ul>
                    </div>
                  </div>
                  <p className="text-sm text-gray-600 mt-3">
                    <strong>The trajectory:</strong> The chart shows historical xwOBA (solid line) up until 2024 and predicted 
                    future performance (shaded area) for 2025. The model accounts for typical aging patterns in baseball.
                  </p>
                </div>
              </CardContent>
            </Card>

            {/* Model Information */}
            <Card>
              <CardHeader>
                <CardTitle>About the Model</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <p className="text-sm text-gray-600">
                    This model uses a transformer neural network trained on Statcast data from 2015-2024. For each player,
                    the model learns how key offensive traits  :- such as <strong> strikeout rate, walk rate, exit velocity and batted ball profile 
                      </strong> -: 
                    evolve over time, and how those traits predict a player's xwOBA.
                    The shaded area represents 
                    uncertainty in the prediction for the 2025 season.
                  </p>
    
            
  
                </div>
              </CardContent>
            </Card>

            {/* What‑if controls */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <SlidersHorizontal size={18} /> What‑if sliders (next season deltas)
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
            
            
                <p className="text-sm text-gray-600 mb-4">
                  <strong>What gets modified:</strong> The sliders let you explore hypothetical scenarios by letting you adjust the player's most recent season's data 
                  (their latest available stats).When you adjust a slider, you're asking: "What if this player's underlying
                   skill changed by this amount?" The model then recalculates their entire future xwOBA projection based on this change. 
                   Think of it like adjusting the inputs to a complex equation and seeing how the output changes.
                </p>
                
                {[
                  { 
                    key: "k", 
                    label: "K % (‑5 → +5)", 
                    description: "Strikeout percentage - how often the player strikes out. Lower is generally better for hitters",
                    range: "Typically 15-30% for MLB hitters",
                    impact: "Lower K% usually means better contact skills and higher xwOBA"
                  },
                  { 
                    key: "bb", 
                    label: "BB % (‑5 → +5)", 
                    description: "Walk percentage - how often the player walks. Higher indicates better plate discipline",
                    range: "Typically 5-15% for MLB hitters",
                    impact: "Higher BB% usually means better plate discipline and higher xwOBA"
                  },
                  { 
                    key: "maxev", 
                    label: "Max EV (‑3 → +3)", 
                    description: "Maximum exit velocity - how hard they can hit the ball. Higher indicates more power potential",
                    range: "Typically 100-120 mph for power hitters",
                    impact: "Higher Max EV usually means more power potential and higher xwOBA"
                  },
                  { 
                    key: "gbfb", 
                    label: "GB/FB (‑1 → +1)", 
                    description: "Ground ball to fly ball ratio - affects power and batting average on balls in play",
                    range: "0.5-2.0, lower values favor power hitters",
                    impact: "Lower GB/FB usually means more fly balls and potentially more power"
                  },
                ].map(({ key, label, description, range, impact }) => (
                  <div key={key} className="space-y-2">
                    <div className="flex justify-between items-start">
                      <div>
                        <p className="font-medium text-sm">{label}</p>
                        <p className="text-xs text-gray-500">{description}</p>
                        <p className="text-xs text-gray-400">{range}</p>
                        <p className="text-xs text-blue-600 font-medium">{impact}</p>
                      </div>
                      <span className="text-sm font-mono bg-gray-100 px-2 py-1 rounded">
                        {whatIf[key].toFixed(2)}
                      </span>
                    </div>
                    <Slider
                      min={key === "gbfb" ? -1 : -5}
                      max={key === "gbfb" ? 1 : key === "maxev" ? 3 : 5}
                      step={0.1}
                      value={[whatIf[key]]}
                      onValueChange={([v]) => setWhatIf({ ...whatIf, [key]: v })}
                      onValueCommit={handleWhatIfCommit}
                    />
                  </div>
                ))}
              </CardContent>
            </Card>

            {/* Explanation panel */}
            <Card>
              <CardHeader>
                <CardTitle>Scouting note</CardTitle>
              </CardHeader>
              <CardContent>
                {loadingExplain ? (
                  <p className="text-sm text-gray-500 animate-pulse">Generating…</p>
                ) : (
                  <>
                    <Button size="sm" onClick={fetchExplanation} className="mb-3">
                      Generate explanation
                    </Button>
                    {explanation && <p className="whitespace-pre-wrap">{explanation}</p>}
                  </>
                )}
              </CardContent>
            </Card>
          </>
        )}
      </div>
    </div>
  );
}
