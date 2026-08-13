import { PredictorClient } from "@/components/PredictorClient";

export const metadata = { title: "Price Predictor | PricePoint Dynamics" };

export default function PredictorPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold text-text-primary">Interactive Price Predictor</h1>
        <p className="mt-2 max-w-2xl text-sm text-text-secondary">
          Search for a product and store to get a real-time price prediction from the trained LightGBM model. Every
          feature except an optional &ldquo;yesterday&apos;s price&rdquo; override is resolved from real history --
          nothing is fabricated (project_refactor.md defect #6).
        </p>
      </div>
      <PredictorClient />
    </div>
  );
}
