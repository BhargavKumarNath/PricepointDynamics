import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import load_canonical_data

st.set_page_config(layout="wide")
st.title("🧺 Basket-Level Price Comparison") 
st.markdown("Here we compare the cost of standardised shopping baskets across supermarkets, powered by our advanced product matching model.")

@st.cache_data
def get_pivot_table():
    """Takes the loaded canonical data and creates the pivot table needed for this page."""
    df = load_canonical_data() 
    latest_date = pd.to_datetime(df['date']).max()
    df_latest = df[df['date'] == latest_date].copy()
    df_latest = df_latest.drop_duplicates(subset=['canonical_name', 'supermarket'])
    pivot = df_latest.pivot_table(index='canonical_name', columns='supermarket', values='prices', observed=True)
    return pivot

price_pivot = get_pivot_table()


baskets = {
    "The Essentials": ['finest goat cheese caramelised red onion ravioli', 'gran luchito mexican crunchy jalapeño pineapple', 'lancashire farm greek style fat free yogurt', 'sma pro follow on baby milk liquid ready to feed', 'mamia organic tomato wheels', 'gallo risotto with tomato and basil', 'bodrum bodrum crispy fried onion', 'lindt classic recipe hazelnut milk chocolate bar', 'no added sugar apple pear juice drink cartons', 'sainsburys british fresh chicken breast pieces in a salt chilli breadcrumb coating', 'president french brie cheese', 'sainsburys cauliflower cheese', 'port salut french creamy cheese slices 6x20g', 'mutti baby roma tomatoes', 'up go banana honey breakfast shake'],
    "The Big Brand Shop": ['kelloggs krave chocolate hazelnut cereal', 'hovis original wheatgerm bread', 'walkers meaty variety crisps 12x25g', 'kelloggs cereal wheats blueberry', 'kelloggs sultana bran', 'heinz apple yoghurt 4 months', 'walkers salt vinegar multipack crisps 6x25g', 'starbucks cappuccino by nescafe d', 'heinz salad cream 70 less fat', 'nescafe dolce gusto cafe au lait coffee x16 pods 16 drinks', 'walkers 45 less salt salted crisps', 'nescafe gold blend instant coffee', 'heinz by nature sweet potato lean beef hotpot 12 months', 'kelloggs rice krispies curious caramel chocolate snack bars', 'walkers baked salt vinegar snacks crisps'],
    "The Healthy Choice": ['grace mango carrot drink', 'telma chicken flavour stock cubes', 'lancashire farm greek style fat free yogurt', 'sainsburys british fresh chicken breast pieces in a salt chilli breadcrumb coating', 'sainsburys cat treat deli cat sticks chicken liver x10', 'meatiful british chicken with brown rice complete dog food', 'oceans halo organic spring roll rice wraps', 'schwartz chargrill chicken seasoning', 'finest 2 cornfed free range chicken breast fillets', 'napolina penne pasta', 'british chicken thigh fillets', 'sainsburys on the go milk chocolate rice cakes', 'sainsburys honey mustard chicken pasta', 'finest purple sprouting broccoli', 'old el paso kit chilli paprika burrito rice'],
    "Food Cupboard": ['dairyfine white chocolate cookies cream bunny', 'just good sauce co chunky burger sauce fully loaded 25', 'pickled red cabbage', 'dominion foam bananas shrimps', 'dairyfine mini chocolate eggs', 'bake by strong white bread flour', 'batchelors big pasta n sauce chicken mushroom flavour pot', 'lindt gold bunny salted caramel milk chocolate', 'broad beans in water', 'dr oetker bright bold sprinkles mix', 'oreo twists vanilla caramel cookie sandwich biscuits', 'finest beef gravy', 'najma spicy breaded chicken fillets', 'habanero hot chilli sauce', 'old el paso cheesy baked enchilada kit', 'taylor colledge organic vanilla bean paste', 'sainsburys best of british apples x6', 'swizzels drumstick squashies banana blueberry', 'quaker chocolate orange porridge pot', 'sainsburys ground allspice'],
    "Health Products": ['colgate max white stain guard toothpaste', 'lacura for men activate shower gel', 'flamingo ibuprofen tablets tablets', 'lacura mens impact 48hr antiperspirant deodorant', 'bells healthcare hayfever allergy relief 10mg tablet', 'max factor miracle pure nail polish nude rose', 'max factor all day flawless 91 warm amber foundation', 'collection filter finish 1 sweet nothings liquid blush light wand', 'carex antibacterial moisture plus hand wash may come in refill pack', 'george home fresher for longer bath towel blue', 'rimmel london super gel nail polish grape sorbet', 'keia nourishing lip balm', 'oralb pro expert healthy white toothpaste', 'head shoulders 2 in 1 classic clean shampoo conditioner', 'maxi nutrition max whey protein powder vanilla', 'natures bounty beauty complex with biotin caplets x60', 'nivea cellular luminous 630 anti dark spot serum', 'hri water balance 30 tablets', 'aveeno fresh greens blend conditioner', 'tena men active fit incontinence pants plus medium x9'],
    "Fresh Food": ['natures pick beansprouts', 'hake fillets', 'acti leaf barista oat drink', 'specially selected aromatic warming moroccan inspired', 'inspired cuisine chicken bacon pasta bake', 'george home blue white pure saute pan 28cm', 'extra special langres aop', 'robinsons creations pineapple mango passion fruit squash', 'black forest fruits mix', 'cadbury twirl milk chocolate orange flavoured bars', 'pilgrims choice lighter extra mature cheddar', 'gu chocolate melting middles 2x100g', 'traditional chicken hotpot', 'muller corner mixed red fruits yogurts', 'market street mackerel fillet', 'yeo valley kefir rhubarb fermented organic yogurt', 'sainsburys french garlic sausage slices x14', 'sainsburys outdoor bred oak smoked dry cured british gammon taste the difference approx', 'sainsburys douro taste the difference 75cl', 'sainsburys danish maple pecan plait x2'],
    "Drinks": ['bishops finger kentish strong ale', 'red bull energy drink 2x250ml', 'tassimo coffee shop classics typ latte', 'old hopking strawberry daiquiri frozen cocktail pouch 2', 'sun quench boost raspberry strawberry acai flavoured', 'extra special sauvignon blanc', 'no added sugar zero tropical crush', 'breville bold vanilla cream 2slice toaster cream silver chrome vtr003', 'twinings the earl grey 120 plantbased tea bags', 'badger tangle foot traditional golden ale', 'the best sancerre', 'pallini limoncello 50cl', 'j2o apple raspberry 4 cans', 'beams the whisky collection 5cl gift set', 'birra moretti lager beer bottles18 x', 'johnnie walker gold label reserve whisky bottle 40 vol 70cl', 'reign mang o matic', 'cune rioja reserva 75cl', 'mighty ultimate barista oat milk alternative long life', 'for goodness shake recovery chocolate flavour'],
    "Household": ['saxon blast kitchen towel 100 sheets 1 roll', 'hotel collection no3 pomegranate fragranced candle 140', 'beautiful blooms fairtrade roses', 'purewick starlight fragranced reed diffuser', 'powerforce antibacterial citrus biodegradable surface w', 'elbow grease limited edition washing up liquid pink blush fragrance', 'george home wipe clean placemat styles may vary', 'auto drive 19 inch universal flat blade', 'ellas kitchen chicken curry with veggie rice', 'duck colouring rimblock green', 'drawstring tall k', 'nutmeg home large jar sandalwood', 'cif antibac shine multipurpose cle', 'lenor fabric conditioner floral fresh 80 washes', 'glade bathroom gel air freshener sandalwood jasmine', 'sainsburys antibacterial total cleaning surface wipes x40', 'bloo original blue toilet blocks 2x50g', 'zoflora concentrated multipurpose disinfectant raspberry juniper berry', 'method multi surface concentrate joyful bing cherry bergamot', 'smash single walled bottle black'],
    "Free From": ['plant menu naked katsu curry', 'barone montalto grillo sicilia doc 75cl', 'polli capers in vinegar capotes', 'plant menu indian inspired no chicken strips', 'cocobay rum and coconut flavour liqueur 70cl', 'tiger tiger 5mm rice sticks noodles', 'richmond meat free vegan 6 honey roasted style slices', 'hartleys 10 cal strawberry jelly', 'free from choc bar', 'pukka chicken bacon slice', 'moo free choccy rocks moofreesas', 'alpro soya chilled drink 1 litre', 'arla lactofree fresh whole milk drink 1 litre', 'oatly oat drink chilled no sugars', 'popworks sweet salty grab bag popped crisps', 'sainsburys creamy peppercorn sauce mix inspired to cook', 'amoy straight to wok medium noodles 2x150g', 'pukka organic ginger galangal golden turmeric tea bags x20', 'pringles hot flamin cheese flavour crisps', 'sainsburys fresh egg lasagne sheets'],
    "Frozen Foods": ['slimfast nourish pots spicy mexican bean chilli', 'everyday essentials chicken breast fillets', 'crestwood garlic herb cheese bites', 'plant menu crispy battered no beef strips', 'ben jerrys ice cream tub non dairy caramel café sundae', 'omv deliciously vegan 2 no beef burgers', 'heinz korma curry frozen bowl ready meal', 'youngs gastro signature breaded 2 sweet chilli fish fillets', 'birds eye steam fresh spinach r', 'birds eye 4 chicken burgers', 'ballineen cooked irish saus', 'chargrilled mixed antipasti', 'stonebaked cheese pizza', 'humza beef cured sliced', 'weight watchers sweet sour chicken', 'birds eye beef quarter pounders with plant protein onion x4', 'sainsburys broccoli cauliflower florets', 'sainsburys classic crust veggie delight pizza', 'quorn roarsomes crunchy dinosaurs', 'ben jerrys caramel chew chew ice cream tub'],
    "Pets": ['earls langhams dog food tray grain free mixed selec', 'vitacat pawsome pockets with cheese', 'earls tender pâté with beef', 'earls mini nibble dog treats', 'langhams meaty chicken breast fillets', 'hero by chicken dry dog food puppy', 'whiskas 1 country collection mix adult wet cat food pouch in gravy', 'pets pantry complete meaty chunks with tasty chicken', 'pedigree dentastix daily adult small dog dental treats 35 sticks', 'chappie complete adult dry dog food beef wholegrain cereal', 'petface puppy pads x100', 'peckish extra goodness suet cake', 'dreamies cat treat biscuits with catnip', 'purina one sensitive dry cat fo', 'gourmet nature creation gravy ckn', 'winalot perfect portions dog food mixed in gravy 40x100g', 'yumove joint care daily tasty bites for adult dogs x60', 'gocat senior chicken turkey mix with vegetables', 'forthglade natural soft bites turkey', 'petface chicken drummer treat toy combo'],
    "Baby Products": ['kiddylicious mini coconut rolls', 'mamia ultrafit xl nappy pants size 6', 'organic mamia organic vegetable beef hotpot tray meal', 'bear paws fruit shapes strawberry apple', 'ellas kitchen pineapple mango orange melty hoops 10', 'aptamil advanced 3 formula toddler milk powder 13 years', 'organix finger foods cheese stars 7 months', 'cow and gate banana wholegrain porridge baby cereal from 7 month', 'mam night 0 months soother', 'childs farm hair body wash organic sweet orange', 'tena incontinence pants plus l', 'weleda baby all purpose balm calendula', 'heinz sweet sour chicken b', 'johnsons baby natural powder', 'ellas kitchen organic pasta bolognese t', 'nuby cup mighty swig', 'large oval cotton wool pads double faced 50', 'organix melty carrot puffs', 'tommee tippee cherry latex soother 1836 months', 'ellas kitchen organic dairy free pear fig porridge baby food pouch 6 months'],
    "Bakery": ['specially selected luxury chocolate brookies 4 pac', 'saint aubert 6 chocolate filled crepes', 'oven bottom muffins', 'dairyfine chocums cookies 4x60g', 'village bakery berry mini cupcakes', 'extra special white chocolate salted caramel cupcakes', 'iced coffee cake', 'schar gluten free white ciabatta rolls', 'the bakery at 6 bramley apple pies', 'thorntons celebration cake each', 'pataks flame baked peshwari naan breads', 'mr kipling chocolate slices', 'pagen gifflar cinnamon', 'finest chocolate caramel tear share brioche', 'cadbury milk chocolate mini rolls cakes x10', 'sainsburys birthday celebration loaded lemon cake taste the difference serves 14', 'sainsburys ancient grain pave taste the difference', 'oat and raisin cookies', 'schar wholesome white loaf', 'flapjack']
}

selected_basket_name = st.selectbox(
    "Choose a shopping basket to analyse:", options=list(baskets.keys())
)
selected_basket_items = baskets[selected_basket_name]

basket_df = price_pivot[price_pivot.index.isin(selected_basket_items)]

items_found = basket_df.notna().sum()
basket_cost = basket_df.sum()
total_items_in_basket = len(selected_basket_items)

summary_df = pd.DataFrame({
    "Total Cost (£)": basket_cost,
    "Items Found": items_found,
    "Coverage (%)": (items_found / total_items_in_basket) * 100
}).sort_values("Total Cost (£)").reset_index()

palette = sns.color_palette("viridis", n_colors=len(summary_df))

fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(data=summary_df, x='supermarket', y='Total Cost (£)', hue='supermarket', palette=palette, ax=ax, legend=False)
ax.set_title(f"Cost of '{selected_basket_name}' Basket", fontsize=18, weight='bold')
ax.set_xlabel("Supermarket", fontsize=12)
ax.set_ylabel("Total Basket Price (£)", fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
st.pyplot(fig)

st.subheader("Detailed Basket Breakdown")
st.markdown("The table below shows the total cost, the number of items found from the basket, and the percentage of the basket each supermarket was able to fulfill.")
st.dataframe(summary_df, width='stretch')


with st.expander("View Products in this Basket and Their Prices"):
    st.markdown(f"**Showing {len(basket_df)} of {total_items_in_basket} products found in the database for the latest date.**")
    st.dataframe(basket_df.style.format("{:.2f}", na_rep="Not Stocked").highlight_min(axis=1, color='#D4EDDA').highlight_max(axis=1, color='#F8D7DA'))

with st.expander("How was this analysis possible? (The Tech Behind It)"):
    st.markdown("""
    A simple text match on product names found only ~3,000 comparable products. To overcome this, I built an advanced pipeline:
    1.  **Text Normalization:** Cleaned and standardized over 127,000 unique product names.
    2.  **Sentence-BERT Embeddings:** Used the `e5-large` transformer model to convert product names into meaningful vector representations.
    3.  **FAISS Similarity Search:** Employed Facebook AI's high-speed vector search library to identify clusters of semantically identical products.
    
    This process expanded our comparable product universe to over **67,000 items**, enabling this robust, multi-category basket analysis.
    """)