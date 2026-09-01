<?php

namespace Database\Seeders;

use App\Models\Violation;
use Illuminate\Database\Seeder;

class ViolationSeeder extends Seeder
{
    /**
     * Seed representative development violations without creating duplicates.
     */
    public function run(): void
    {
        $violations = [
            [
                'event_id' => 'DEMO_OVERSPEED_001',
                'track_id' => 101,
                'plate_number' => 'ঢাকা মেট্রো গ 12-3456',
                'speed' => 82.40,
                'speed_limit' => 60.00,
                'violation_type' => 'OVERSPEED',
                'signal_state' => 'GREEN',
                'direction' => 'UP',
                'vehicle_color' => 'RED',
                'color_confidence' => 0.84,
                'frame_number' => 1250,
                'frame_timestamp' => now()->subMinutes(35),
                'ocr_raw_text' => 'ঢাকা মেট্রো গ ১২-৩৪৫৬',
                'ocr_confidence' => 0.91,
                'ocr_engine' => 'easyocr',
            ],
            [
                'event_id' => 'DEMO_RED_LIGHT_002',
                'track_id' => 102,
                'plate_number' => 'চট্টগ্রাম মেট্রো চ 11-7788',
                'speed' => 43.70,
                'speed_limit' => null,
                'violation_type' => 'RED_LIGHT',
                'signal_state' => 'RED',
                'direction' => 'DOWN',
                'vehicle_color' => 'WHITE',
                'color_confidence' => 0.78,
                'frame_number' => 1480,
                'frame_timestamp' => now()->subMinutes(18),
                'ocr_raw_text' => 'চট্টগ্রাম মেট্রো চ ১১-৭৭৮৮',
                'ocr_confidence' => 0.87,
                'ocr_engine' => 'easyocr',
            ],
            [
                'event_id' => 'DEMO_UNREADABLE_003',
                'track_id' => 103,
                'plate_number' => 'UNREADABLE',
                'speed' => null,
                'speed_limit' => null,
                'violation_type' => 'RED_LIGHT',
                'signal_state' => 'RED',
                'direction' => 'UP',
                'vehicle_color' => 'BLACK',
                'color_confidence' => 0.66,
                'frame_number' => 1655,
                'frame_timestamp' => now()->subMinutes(5),
                'ocr_raw_text' => null,
                'ocr_confidence' => 0.0,
                'ocr_engine' => 'none',
            ],
        ];

        foreach ($violations as $violation) {
            Violation::updateOrCreate(
                ['event_id' => $violation['event_id']],
                $violation,
            );
        }
    }
}
