<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    /**
     * Run the migrations.
     */
    public function up(): void
    {
        Schema::create('violations', function (Blueprint $table) {
            $table->id();
            $table->string('event_id')->unique();
            $table->unsignedBigInteger('track_id')->nullable();
            $table->string('plate_number')->default('UNREADABLE');
            $table->decimal('speed', 7, 2)->nullable();
            $table->decimal('speed_limit', 7, 2)->nullable();
            $table->string('violation_type', 32);
            $table->string('signal_state', 16)->nullable();
            $table->string('direction', 16)->nullable();
            $table->unsignedBigInteger('frame_number');
            $table->timestamp('frame_timestamp', precision: 3);
            $table->string('image_path')->nullable();
            $table->string('plate_crop_path')->nullable();
            $table->text('ocr_raw_text')->nullable();
            $table->decimal('ocr_confidence', 6, 5)->nullable();
            $table->string('ocr_engine', 32)->nullable();
            $table->timestamps();

            $table->index('plate_number');
            $table->index('violation_type');
            $table->index('created_at');
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('violations');
    }
};
