<?php

namespace App\Models;

use Database\Factories\ViolationFactory;
use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;

class Violation extends Model
{
    /** @use HasFactory<ViolationFactory> */
    use HasFactory;

    /**
     * Attributes accepted from the Python violation pipeline and dashboard.
     *
     * @var list<string>
     */
    protected $fillable = [
        'event_id',
        'track_id',
        'plate_number',
        'original_plate_number',
        'plate_corrected_at',
        'speed',
        'speed_limit',
        'violation_type',
        'signal_state',
        'direction',
        'vehicle_color',
        'color_confidence',
        'frame_number',
        'frame_timestamp',
        'image_path',
        'plate_crop_path',
        'ocr_raw_text',
        'ocr_confidence',
        'ocr_engine',
        'status',
        'reviewed_by',
        'reviewed_at',
        'officer_notes',
    ];

    /**
     * Get the attributes that should be cast.
     *
     * @return array<string, string>
     */
    protected function casts(): array
    {
        return [
            'track_id' => 'integer',
            'speed' => 'float',
            'speed_limit' => 'float',
            'color_confidence' => 'float',
            'frame_number' => 'integer',
            'frame_timestamp' => 'datetime',
            'plate_corrected_at' => 'datetime',
            'ocr_confidence' => 'float',
            'reviewed_at' => 'datetime',
        ];
    }

    public function reviewer(): BelongsTo
    {
        return $this->belongsTo(User::class, 'reviewed_by');
    }
}
