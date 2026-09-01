<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public function up(): void
    {
        Schema::table('violations', function (Blueprint $table) {
            $table->string('status', 20)->default('PENDING')->after('ocr_engine');
            $table->string('original_plate_number')->nullable()->after('plate_number');
            $table->timestamp('plate_corrected_at', precision: 3)->nullable()->after('original_plate_number');
            $table->foreignId('reviewed_by')->nullable()->after('status')
                ->constrained('users')->nullOnDelete();
            $table->timestamp('reviewed_at', precision: 3)->nullable()->after('reviewed_by');
            $table->text('officer_notes')->nullable()->after('reviewed_at');

            $table->index('status');
            $table->index('reviewed_at');
        });
    }

    public function down(): void
    {
        Schema::table('violations', function (Blueprint $table) {
            $table->dropForeign(['reviewed_by']);
            $table->dropIndex(['status']);
            $table->dropIndex(['reviewed_at']);
            $table->dropColumn([
                'status',
                'original_plate_number',
                'plate_corrected_at',
                'reviewed_by',
                'reviewed_at',
                'officer_notes',
            ]);
        });
    }
};
