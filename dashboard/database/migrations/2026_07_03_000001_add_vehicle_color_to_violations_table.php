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
        Schema::table('violations', function (Blueprint $table) {
            $table->string('vehicle_color', 24)
                ->default('UNKNOWN')
                ->after('direction');
            $table->decimal('color_confidence', 6, 5)
                ->nullable()
                ->after('vehicle_color');

            $table->index('vehicle_color');
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::table('violations', function (Blueprint $table) {
            $table->dropIndex(['vehicle_color']);
            $table->dropColumn(['vehicle_color', 'color_confidence']);
        });
    }
};
