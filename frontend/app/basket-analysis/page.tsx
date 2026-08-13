import { BasketAnalysisClient } from "@/components/BasketAnalysisClient";
import { loadBasketAnalysis } from "@/lib/artifacts";

export const metadata = { title: "Basket Analysis | PricePoint Dynamics" };

export default async function BasketAnalysisPage() {
  const { baskets } = await loadBasketAnalysis();

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold text-text-primary">Basket-Level Price Comparison</h1>
        <p className="mt-2 max-w-2xl text-sm text-text-secondary">
          Compare the cost of standardised shopping baskets across supermarkets, powered by the product matching
          model.
        </p>
      </div>
      <BasketAnalysisClient baskets={baskets} />
    </div>
  );
}
