<?php

namespace Database\Factories;

use App\Models\Violation;
use Illuminate\Database\Eloquent\Factories\Factory;

/**
 * @extends Factory<Violation>
 */
class ViolationFactory extends Factory
{
    protected $model = Violation::class;

    public function definition(): array
    {
        $type = fake()->randomElement(['OVERSPEED', 'RED_LIGHT']);

        return [
            'event_id' => fake()->unique()->uuid(),
            'track_id' => fake()->numberBetween(1, 500),
            'plate_number' => fake()->bothify('DHAKA-??-####'),
            'speed' => $type === 'OVERSPEED' ? fake()->randomFloat(2, 61, 130) : null,
            'speed_limit' => $type === 'OVERSPEED' ? 60 : null,
            'violation_type' => $type,
            'signal_state' => $type === 'RED_LIGHT' ? 'RED' : 'GREEN',
            'direction' => fake()->randomElement(['UP', 'DOWN']),
            'vehicle_color' => fake()->randomElement(['BLACK', 'WHITE', 'GRAY', 'RED']),
            'color_confidence' => fake()->randomFloat(4, 0.4, 0.99),
            'frame_number' => fake()->numberBetween(1, 10000),
            'frame_timestamp' => fake()->dateTimeBetween('-30 days'),
            'ocr_confidence' => fake()->randomFloat(4, 0.35, 0.99),
            'ocr_engine' => 'easyocr',
            'status' => 'PENDING',
        ];
    }
}
