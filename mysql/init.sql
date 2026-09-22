CREATE DATABASE IF NOT EXISTS docsearch
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

CREATE USER IF NOT EXISTS 'docsearch'@'localhost' IDENTIFIED BY 'docsearch';
CREATE USER IF NOT EXISTS 'docsearch'@'127.0.0.1' IDENTIFIED BY 'docsearch';
GRANT ALL PRIVILEGES ON docsearch.* TO 'docsearch'@'localhost';
GRANT ALL PRIVILEGES ON docsearch.* TO 'docsearch'@'127.0.0.1';
FLUSH PRIVILEGES;

USE docsearch;

CREATE TABLE IF NOT EXISTS records (
  id INT UNSIGNED NOT NULL AUTO_INCREMENT,
  title VARCHAR(512) NOT NULL,
  content TEXT NOT NULL,
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
    ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

INSERT INTO records (title, content)
SELECT * FROM (
  SELECT 'Helium Bakery Robots Open in Reykjavik' AS title,
         'A downtown bakery installed helium-cooled pastry robots that keep laminated dough at a constant minus two degrees while folding it. Night bakers say the machines hum in a narrow band that also keeps the proofing cabinets frost-free.' AS content
  UNION ALL SELECT 'City Bicycle Ledger Still Uses Paper Stamps',
         'The municipal bicycle-share office records every checkout with a dated paper stamp in a bound ledger. Staff refused a cloud app because winter outages left riders unable to return bikes last year.'
  UNION ALL SELECT 'Subsea Kelp Fiber Carries Winter Power',
         'Engineers floated a kelp-derived fiber cable between two fjord substations. The wet fiber stays flexible in ice and is meant to carry village loads when the overland line ices over.'
  UNION ALL SELECT 'Otters Inspect Sluice Gates on Night Shift',
         'A river authority trained a small team of otters to tug marker rings on sluice hinges after dark. Handlers log each inspection in a waterproof notebook rather than a phone, which fogs in the spray.'
  UNION ALL SELECT 'Ceramic Postage Melts in Rain as Anti-Fraud',
         'The island post office fired thin ceramic stamps that dissolve into chalk when soaked. A letter that arrives with an intact stamp could not have been steamed open in transit.'
  UNION ALL SELECT 'Thunderstorm Archive Prices Crop Insurance',
         'An inland co-op stores labeled recordings of local thunderstorms and plays them back to score hail risk. Underwriters listen for the same crackle pattern before they set the next season''s premium.'
  UNION ALL SELECT 'Glass Harmonica Keeps the Factory Clock',
         'A glassworks replaced its electric shift bell with a glass harmonica that sounds every quarter hour. The wet-finger pitch stays stable when the furnace room loses mains power.'
  UNION ALL SELECT 'Desert Tram Harvests Static for Lamps',
         'A short tram line in the salt flats trails a comb that strips static from the rails and feeds streetlamps at each stop. Riders notice a faint ozone smell only on the driest afternoons.'
  UNION ALL SELECT 'Fermented Ink Flags Expired Contracts',
         'A notary uses fermented oak-gall ink that shifts from brown to green when a stored contract passes its written expiry date. Clerks check the color before they accept a filing.'
  UNION ALL SELECT 'Balloon Post Links Mountain Clinics',
         'Two high clinics exchange lab slips by tethered balloon when the switchback road is closed. Each capsule carries a paper card and a dried-ink blot used to confirm the sender.'
  UNION ALL SELECT 'Salt-Block Servers Run Village Weather',
         'A coastal village stacked salt-block computers in a shed to run a three-kilometer weather model. The blocks wick heat and need only a hand-crank fan during the afternoon glare.'
  UNION ALL SELECT 'Tuning-Fork Choir Encodes Tide Tables',
         'Harbor pilots keep a rack of stamped tuning forks whose beat notes encode the week''s tide heights. A newcomer learns the intervals instead of unfolding a wet paper chart on deck.'
) AS seed
WHERE NOT EXISTS (SELECT 1 FROM records LIMIT 1);
